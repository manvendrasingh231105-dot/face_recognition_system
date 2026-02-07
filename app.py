from flask import Flask, render_template, Response, jsonify, send_file, request
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import csv
import os
import sys

app = Flask(__name__)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --------- Camera ----------
camera = cv2.VideoCapture(0)

# --------- Attendance CSV ----------
attendance_file = "attendance.csv"
attendance_set = set()

if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])

# --------- Face Embeddings Storage ----------
EMBED_DIR = "embeddings"
os.makedirs(EMBED_DIR, exist_ok=True)


def load_known_faces():
    faces = {}
    for fname in os.listdir(EMBED_DIR):
        if fname.endswith(".npy"):
            path = os.path.join(EMBED_DIR, fname)
            name = os.path.splitext(fname)[0]      # file name without .npy
            emb = np.load(path)
            faces[name] = emb
    return faces


KNOWN_FACES = load_known_faces()  # name -> embedding vector


def extract_embedding(landmarks_obj):
    """Convert MediaPipe landmarks into a 1D embedding vector."""
    pts = []
    for lm in landmarks_obj.landmark:
        pts.extend([lm.x, lm.y, lm.z])
    return np.array(pts, dtype=np.float32)


def recognize_face(embedding):
    """Compare embedding with stored ones and return best match name."""
    if not KNOWN_FACES:
        return "Unknown"

    best_name = "Unknown"
    best_dist = 1e9

    for name, known_emb in KNOWN_FACES.items():
        if known_emb.shape != embedding.shape:
            continue
        dist = np.linalg.norm(known_emb - embedding)
        if dist < best_dist:
            best_dist = dist
            best_name = name

    # Threshold – you can tweak this if needed
    if best_dist < 1.2:
        return best_name
    return "Unknown"


def generate_frames():
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        name = "No Face"

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            embedding = extract_embedding(landmarks)
            name = recognize_face(embedding)

            # Mark attendance if recognized and not already marked
            display_name = name
            if name != "Unknown":
                # if you used underscores in filenames, show with spaces
                display_name = name.replace("_", " ")

                if display_name not in attendance_set:
                    attendance_set.add(display_name)
                    with open(attendance_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([display_name, datetime.now().strftime("%H:%M:%S")])

            cv2.putText(frame, display_name, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "No Face", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/get_attendance")
def get_attendance():
    with open(attendance_file, "r") as file:
        data = list(csv.reader(file))
    return jsonify(data)


@app.route("/download")
def download():
    return send_file(attendance_file, as_attachment=True)


# ---------- NEW: Register Student Face ----------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    raw_name = data.get("name", "").strip()
    if not raw_name:
        return jsonify({"success": False, "message": "Name is required"}), 400

    # Use a safe file-friendly ID (no spaces)
    safe_name = raw_name.replace(" ", "_")

    # Capture one frame from webcam
    ret, frame = camera.read()
    if not ret:
        return jsonify({"success": False, "message": "Camera error"}), 500

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return jsonify({"success": False, "message": "No face detected. Try again."}), 400

    landmarks = results.multi_face_landmarks[0]
    embedding = extract_embedding(landmarks)

    # Save embedding to file
    path = os.path.join(EMBED_DIR, f"{safe_name}.npy")
    np.save(path, embedding)

    # Update in-memory dict so restart not required
    KNOWN_FACES[safe_name] = embedding

    return jsonify({
        "success": True,
        "message": f"Registered {raw_name} successfully. Look at the camera similarly for recognition."
    })


# ---------- Shutdown ----------
@app.route("/shutdown", methods=["POST"])
def shutdown():
    try:
        camera.release()
        cv2.destroyAllWindows()
        func = request.environ.get("werkzeug.server.shutdown")
        if func:
            func()
        return "Server shutting down..."
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    app.run(debug=True)
