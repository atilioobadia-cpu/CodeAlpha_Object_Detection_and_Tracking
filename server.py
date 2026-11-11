from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 150 * 1024 * 1024
MODEL_NAME = os.getenv("VISION_MODEL", "yolo26n.pt")
TRACKER_NAME = "bytetrack.yaml"
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = Flask(__name__, template_folder=".", static_folder=".")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

LOCK = threading.RLock()
MODEL_LOCK = threading.Lock()
model: Any = None
run_state: dict[str, Any] = {
    "status": "idle",
    "source": "",
    "active": False,
    "fps": 0.0,
    "frames": 0,
    "objects": 0,
    "classes": {},
    "error": "",
    "latest_frame": None,
    "stop_event": None,
    "thread": None,
    "capture": None,
}


def get_model():
    """Load the pretrained model on demand; Ultralytics caches its weights."""
    global model
    if model is None:
        with MODEL_LOCK:
            if model is None:
                from ultralytics import YOLO

                logging.info("Loading pretrained model %s", MODEL_NAME)
                model = YOLO(MODEL_NAME)
    return model


def current_status() -> dict[str, Any]:
    with LOCK:
        return {
            key: run_state[key]
            for key in ("status", "source", "active", "fps", "frames", "objects", "classes", "error")
        } | {"model": MODEL_NAME, "tracker": "ByteTrack"}


def tracked_objects(result: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], {}

    coordinates = boxes.xyxy.int().cpu().tolist()
    classes = boxes.cls.int().cpu().tolist()
    confidences = boxes.conf.float().cpu().tolist()
    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(coordinates)
    objects = []
    for xyxy, class_id, confidence, track_id in zip(coordinates, classes, confidences, ids):
        label = str(result.names.get(class_id, class_id))
        objects.append({"id": track_id, "label": label, "confidence": round(float(confidence), 2), "box": xyxy})
    return objects, dict(Counter(item["label"] for item in objects))


def set_run_state(**values: Any) -> None:
    with LOCK:
        run_state.update(values)


def _read_frame_with_timeout(capture: Any, timeout: float = 3.0) -> tuple[bool, Any, bool]:
    """Read one frame without blocking forever. Returns (success, frame, timed_out)."""
    result: dict[str, Any] = {"success": False, "frame": None, "timed_out": False}

    def worker() -> None:
        try:
            success, frame = capture.read()
            result["success"] = bool(success)
            result["frame"] = frame
        except Exception as error:  # pragma: no cover - defensive
            result["error"] = str(error)

    thread = threading.Thread(target=worker, daemon=True, name="camera-read-probe")
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        result["timed_out"] = True
        try:
            capture.release()
        except Exception:
            pass
        thread.join(timeout=1.0)
    return bool(result["success"]), result["frame"], bool(result["timed_out"])


def _open_capture(source: int | str) -> tuple[Any, Any]:
    """Open a capture and verify it can return at least one frame."""
    if isinstance(source, int) and os.name == "nt":
        backends: list[int | None] = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
    elif isinstance(source, int):
        backends = [None]
    else:
        backends = [None]

    for backend in backends:
        capture = cv2.VideoCapture(source, backend) if backend is not None else cv2.VideoCapture(source)
        if not capture.isOpened():
            try:
                capture.release()
            except Exception:
                pass
            continue

        if isinstance(source, int):
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        success, frame, timed_out = _read_frame_with_timeout(capture)
        if timed_out:
            try:
                capture.release()
            except Exception:
                pass
            continue
        if success and frame is not None:
            return capture, frame
        try:
            capture.release()
        except Exception:
            pass

    if isinstance(source, int):
        raise RuntimeError(
            "Could not read from the webcam. Check that a camera is connected, not in use by another app, "
            "and that camera access is allowed. Tried DirectShow, Media Foundation, and the default backend."
        )
    raise RuntimeError("Could not open the video file. Check that it is a valid, readable video.")


def process_stream(source: int | str, source_name: str, confidence: float, stop_event: threading.Event, temp_path: str | None = None) -> None:
    capture = None
    first_frame = None
    try:
        capture, first_frame = _open_capture(source)

        with LOCK:
            run_state["capture"] = capture

        set_run_state(status="loading_model", active=True, source=source_name, error="", fps=0.0, frames=0, objects=0, classes={}, latest_frame=None)
        detector = get_model()
        set_run_state(status="running")
        started_at = time.perf_counter()
        last_fps_at = started_at
        frames_since_fps = 0
        current_fps = 0.0
        frame_count = 0
        pending_frame = first_frame

        while not stop_event.is_set():
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
                success = True
            else:
                success, frame = capture.read()
            if not success or frame is None:
                set_run_state(status="ended" if not isinstance(source, int) else "error", active=False)
                if isinstance(source, int):
                    set_run_state(error="The webcam stopped returning frames. Try starting it again.")
                break

            height, width = frame.shape[:2]
            if width > 1280:
                frame = cv2.resize(frame, (1280, int(height * 1280 / width)), interpolation=cv2.INTER_AREA)

            result = detector.track(
                frame,
                persist=True,
                tracker=TRACKER_NAME,
                conf=confidence,
                imgsz=480,
                verbose=False,
            )[0]
            annotated = result.plot(line_width=2, font_size=0.55)
            objects, class_counts = tracked_objects(result)
            encoded_ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if encoded_ok:
                frames_since_fps += 1
                now = time.perf_counter()
                if now - last_fps_at >= 1.0:
                    current_fps = frames_since_fps / (now - last_fps_at)
                    frames_since_fps = 0
                    last_fps_at = now
                frame_count += 1
                set_run_state(
                    status="running",
                    active=True,
                    fps=round(current_fps, 1),
                    frames=frame_count,
                    objects=len(objects),
                    classes=class_counts,
                    latest_frame=encoded.tobytes(),
                )
    except Exception as error:  # Keep the web interface available if the model or camera fails.
        logging.exception("Video processing stopped")
        set_run_state(status="error", active=False, error=str(error))
    finally:
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                logging.warning("Could not remove temporary video %s", secure_filename(Path(temp_path).name))
        with LOCK:
            run_state["active"] = False
            run_state["stop_event"] = None
            run_state["thread"] = None
            run_state["capture"] = None
            if run_state["status"] in {"running", "loading_model", "starting"}:
                if run_state["status"] != "error":
                    run_state["status"] = "stopped"


def start_stream(source: int | str, source_name: str, confidence: float, temp_path: str | None = None):
    with LOCK:
        if run_state["active"]:
            return jsonify({"error": "A video is already being processed. Stop it before starting another source."}), 409
        stop_event = threading.Event()
        run_state.update(
            status="starting",
            source=source_name,
            active=True,
            fps=0.0,
            frames=0,
            objects=0,
            classes={},
            error="",
            latest_frame=None,
            stop_event=stop_event,
            capture=None,
        )
        worker = threading.Thread(
            target=process_stream,
            args=(source, source_name, confidence, stop_event, temp_path),
            daemon=True,
            name="object-detection-stream",
        )
        run_state["thread"] = worker
        worker.start()
    return jsonify({"status": "starting", "source": source_name}), 202


def read_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.35
    return max(0.1, min(confidence, 0.9))


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/styles.css")
def styles():
    return app.send_static_file("styles.css")


@app.get("/app.js")
def script():
    return app.send_static_file("app.js")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "tracker": "ByteTrack"})


@app.get("/api/status")
def status():
    return jsonify(current_status())


@app.post("/api/start")
def start_camera():
    payload = request.get_json(silent=True) or {}
    return start_stream(0, "Webcam", read_confidence(payload.get("confidence")))


@app.post("/api/upload")
def upload_video():
    video = request.files.get("video")
    if video is None or not video.filename:
        return jsonify({"error": "Choose a video file first."}), 400
    safe_name = secure_filename(video.filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        return jsonify({"error": "Choose an MP4, MOV, AVI, MKV, or WEBM video."}), 400
    with LOCK:
        if run_state["active"]:
            return jsonify({"error": "Stop the current stream before uploading another video."}), 409
    with tempfile.NamedTemporaryFile(prefix="codealpha-video-", suffix=suffix, delete=False) as temporary:
        temp_path = temporary.name
    video.save(temp_path)
    return start_stream(temp_path, Path(safe_name).name, read_confidence(request.form.get("confidence")), temp_path)


@app.post("/api/stop")
def stop_stream():
    with LOCK:
        stop_event = run_state.get("stop_event")
        worker = run_state.get("thread")
        capture = run_state.get("capture")
        if not stop_event or not run_state["active"]:
            return jsonify(current_status())
        run_state["status"] = "stopping"
        stop_event.set()
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
    if worker and worker is not threading.current_thread():
        worker.join(timeout=3)
        if worker.is_alive():
            # Last resort: releasing again cannot hurt; report current state.
            pass
    return jsonify(current_status())


@app.get("/api/stream")
def stream():
    def generate():
        while True:
            with LOCK:
                frame = run_state.get("latest_frame")
                active = run_state["active"]
                status_value = run_state["status"]
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-store\r\n\r\n" + frame + b"\r\n"
            elif not active and status_value in {"idle", "stopped", "ended", "error"}:
                time.sleep(0.2)
            else:
                time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "This video is larger than 150 MB. Choose a smaller file."}), 413


if __name__ == "__main__":
    port = int(os.getenv("PORT", "4176"))
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)