from flask import Flask, render_template, Response, jsonify, request, session

import cv2
import numpy as np
import threading
import time
import os
import atexit
from threading import Lock
import queue
from sort.sort import Sort
app = Flask(__name__)
lock = Lock()


# Inisialisasi tracker SORT global
tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)

# --- Inisialisasi variabel global di luar route ---
available_cameras = [0, 1, 2]  # contoh daftar kamera
selected_camera = 0             # kamera default yang dipilih
max_people_limit = 0
max_weight_limit = 0

# Contoh variabel global (bisa kamu sesuaikan dengan data real dari deteksi)
people_count = 0
total_weight = 0.0
face_count = 0
status = "Normal"

# Global variable untuk VideoCapture
cap = None
capture_thread_started = False 

# Queue untuk buffer frame terbaru (maxsize=1)
frame_queue = queue.Queue(maxsize=1)

# Interval proses frame (detik)
PROCESS_INTERVAL = 0.1

# Threshold confidence dan NMS
CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4

# Filter smoothing (eksponensial)
ALPHA = 0.4

# Variabel global hasil deteksi terakhir
last_total_weight = 0.0
last_face_count = 0
last_count = 0

# File model yang dibutuhkan
required_files = [
    'model/jumlah_orang/yolov3-tiny.weights',
    'model/jumlah_orang/yolov3-tiny.cfg',
    'model/jumlah_orang/coco.names',
    'model/wajah/deploy.prototxt',
    'model/wajah/res10_300x300_ssd_iter_140000.caffemodel',
    'model/berat_badan/age_deploy.prototxt',
    'model/berat_badan/age_net.caffemodel',
]

for path in required_files:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

# Load model YOLO untuk deteksi orang
net = cv2.dnn.readNet('model/jumlah_orang/yolov3-tiny.weights', 'model/jumlah_orang/yolov3-tiny.cfg')
with open('model/jumlah_orang/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load face detector (OpenCV DNN)
face_net = cv2.dnn.readNetFromCaffe('model/wajah/deploy.prototxt', 'model/wajah/res10_300x300_ssd_iter_140000.caffemodel')
face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load age estimation model
age_net = cv2.dnn.readNetFromCaffe('model/berat_badan/age_deploy.prototxt', 'model/berat_badan/age_net.caffemodel')
age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
AGE_LIST = [required_files]

def detect_faces_dnn(person_roi):
    """
    Deteksi wajah pada ROI orang yang sudah di-crop.
    Mengembalikan list bounding box wajah.
    """
    if person_roi is None or person_roi.size == 0:
        return []
    blob = cv2.dnn.blobFromImage(person_roi, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    h, w = person_roi.shape[:2]
    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                results.append((x1, y1, x2 - x1, y2 - y1))
    return results

def estimate_age(face_img):
    """
    Estimasi usia berdasarkan gambar wajah yang sudah di-crop.
    """
    try:
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.426, 87.769, 114.896), swapRB=False)
        age_net.setInput(blob)
        preds = age_net.forward()
        i = preds[0].argmax()
        age_range = AGE_LIST[i]
        low, high = map(int, age_range[1:-1].split('-'))
        return (low + high) / 2
    except Exception as e:
        print(f"[ERROR] Estimasi usia gagal: {e}")
        return 20
    
def estimate_weight_real(age):
    # data = [
    #     (0.5, 7.5), (1, 9.6), (2, 12.2), (3, 14.3), (4, 16.3), (5, 18.3), (6, 20.3), (7, 22.3),
    #     (8, 24.3), (9, 26.3), (10, 28.2), (11, 31.5), (12, 34.0), (13, 38.5), (14, 42.0), (15, 47.0),
    #     (16, 51.5), (17, 56.0), (18, 65.0), (40, 70.0), (60, 70.0)
    # ]
    data = [required_files]
    
    if age <= 0.5:
        return 7.5
    if age >= 60:
        return 70.0
    for i in range(len(data) - 1):
        age1, w1 = data[i]
        age2, w2 = data[i+1]
        if age1 <= age <= age2:
            return w1 + (age - age1)/(age2 - age1) * (w2 - w1)
    return 70.0



def process_frame(frame):
    global last_total_weight, last_count, last_face_count
    global people_count, total_weight, face_count

    if frame is None or frame.size == 0:
        last_total_weight, last_count, last_face_count = 0.0, 0, 0
        people_count, total_weight, face_count = 0, 0.0, 0
        return frame, 0, 0.0, 0

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)

    boxes = []
    confidences = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            if len(scores) == 0:
                continue

            class_id = np.argmax(scores)
            if class_id >= len(classes):
                continue

            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD and classes[class_id] == "person":
                center_x, center_y, width_box, height_box = detection[:4] * np.array([w, h, w, h])
                x = int(center_x - width_box / 2)
                y = int(center_y - height_box / 2)

                x = max(0, x)
                y = max(0, y)
                width_box = min(w - x, int(width_box))
                height_box = min(h - y, int(height_box))

                if width_box < 30 or height_box < 30:
                    continue

                boxes.append([x, y, x + width_box, y + height_box])
                confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    if len(indexes) == 0:
        last_total_weight, last_count, last_face_count = 0.0, 0, 0
        people_count, total_weight, face_count = 0, 0.0, 0
        return frame, 0, 0.0, 0

    indexes = indexes.flatten()
    dets = []
    for i in indexes:
        x1, y1, x2, y2 = boxes[i]
        score = confidences[i]
        dets.append([x1, y1, x2, y2, score])
    dets = np.array(dets)

    tracked_objects = tracker.update(dets)

    total_weight = 0.0
    count = 0
    faces_total = 0

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj.astype(int)
        roi = frame[y1:y2, x1:x2]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Kotak biru untuk tubuh

        faces = detect_faces_dnn(roi)
        face_count_roi = len(faces)
        faces_total += face_count_roi

        ages = []
        for (fx, fy, fw, fh) in faces:
            face_img = roi[fy:fy+fh, fx:fx+fw]
            if face_img.shape[0] < 40 or face_img.shape[1] < 40:
                continue
            resized = cv2.resize(face_img, (227, 227))
            ages.append(estimate_age(resized))
            cv2.rectangle(roi, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 1)  # Kotak wajah kuning

        age = (sum(ages) / len(ages)) if ages else 20
        weight = estimate_weight_real(age)

        label = f"ID:{int(obj_id)} Usia:{int(age)} Berat:{weight:.1f}kg Wajah:{face_count_roi}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        total_weight += weight
        count += 1

    cv2.putText(frame, f"Tubuh Terdeteksi: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Smoothing dengan exponential moving average
    last_total_weight = ALPHA * last_total_weight + (1 - ALPHA) * total_weight
    last_count = ALPHA * last_count + (1 - ALPHA) * count
    last_face_count = ALPHA * last_face_count + (1 - ALPHA) * faces_total

    # Sinkronisasi nilai global untuk API
    people_count = int(last_count)
    total_weight = last_total_weight
    face_count = int(last_face_count)

    return frame, people_count, total_weight, face_count

def gen():
    global frame_detected, count_person, total_weight, count_face, lock

    cap = cv2.VideoCapture(0)  # Ganti dengan sumber video kamu jika perlu

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Proses frame, dapatkan hasil smoothing
        frame_proc, cnt, weight, faces = process_frame(frame)

        # Update variabel global secara thread-safe
        with lock:
            frame_detected = frame_proc.copy()
            count_person = cnt
            total_weight = weight
            count_face = faces

        # Encode frame ke JPEG
        ret2, jpeg = cv2.imencode('.jpg', frame_proc)
        if not ret2:
            continue

        # Hasilkan byte frame MJPEG
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


def capture_frames():
    """
    Thread untuk capture frame dari kamera secara terus-menerus dan memasukkan ke antrian.
    """
    global cap
    frame_count = 0

    while True:
        with lock:
            # Jika kamera belum diinisialisasi atau tidak terbuka
            if cap is None or not cap.isOpened():
                print("[WARNING] Kamera belum tersedia atau tidak terbuka.")
                time.sleep(0.5)
                continue

            ret, frame = cap.read()

        if not ret or frame is None or frame.size == 0:
            print("[WARNING] Gagal membaca frame dari kamera.")
            time.sleep(0.1)
            continue

        # Kosongkan queue agar selalu berisi frame terbaru (maksimal 1)
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            print("[WARNING] Queue penuh saat memasukkan frame.")
            continue

        frame_count += 1
        if frame_count % 30 == 0:   
            print(f"[DEBUG] Frames captured: {frame_count}")

        time.sleep(0.01)


@app.route('/', methods=['GET', 'POST'])
def index():
    global selected_camera, max_people_limit, max_weight_limit, cap

    message = ""
    if request.method == 'POST':
        # Ambil data kamera dari form, ubah tipe ke int
        camera_form_value = request.form.get('camera')
        max_people_val = request.form.get('max_people')
        max_weight_val = request.form.get('max_weight')

        try:
            new_selected_camera = int(camera_form_value) if camera_form_value is not None else selected_camera
            new_max_people_limit = int(max_people_val) if max_people_val is not None else max_people_limit
            new_max_weight_limit = float(max_weight_val) if max_weight_val is not None else max_weight_limit

            with lock:
                # Update kamera jika berubah
                if new_selected_camera != selected_camera:
                    print(f"[INFO] Mengganti kamera dari {selected_camera} ke {new_selected_camera}")
                    selected_camera = new_selected_camera
                    # Release kamera lama dan buka kamera baru
                    if cap is not None:
                        cap.release()
                    cap = cv2.VideoCapture(selected_camera)
                    if not cap.isOpened():
                        message = f"Tidak bisa membuka kamera {selected_camera}"
                        print(f"[ERROR] {message}")

                # Update batas maksimal
                max_people_limit = new_max_people_limit
                max_weight_limit = new_max_weight_limit

            message = "Pengaturan berhasil disimpan."
        except ValueError:
            message = "Input tidak valid. Pastikan angka yang dimasukkan benar."

    return render_template(
        'index.html',
        cams=available_cameras,
        sel=selected_camera,
        max_people=max_people_limit,
        max_weight=max_weight_limit,
        message=message
    )
@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame') 

@app.route('/api/status')
def api_status():
    global people_count, total_weight, face_count, max_people_limit, max_weight_limit

    with lock:
        current_people = int(people_count)
        current_weight = float(total_weight)
        current_faces = int(face_count)
        current_status = "OVERLOAD" if (current_people > max_people_limit or current_weight > max_weight_limit) else "NORMAL"

    return jsonify({
        "people_count": current_people,
        "total_weight": round(current_weight, 2),
        "face_count": current_faces,
        "status": current_status,
        "max_people_limit": max_people_limit,
        "max_weight_limit": max_weight_limit
    })


@app.route('/api/set_limits', methods=['POST'])
def set_limits():
    global max_people_limit, max_weight_limit

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    max_people = data.get("max_people")
    max_weight = data.get("max_weight")

    if not isinstance(max_people, int) or not isinstance(max_weight, (int, float)):
        return jsonify({"error": "Invalid data types"}), 400

    max_people_limit = max_people
    max_weight_limit = max_weight

    return jsonify({
        "message": "Limits updated",
        "max_people_limit": max_people_limit,
        "max_weight_limit": max_weight_limit
    })

def initialize_camera():
    """
    Fungsi untuk inisialisasi kamera pada start aplikasi.
    """
    global cap, selected_camera
    with lock:
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(selected_camera)
        if not cap.isOpened():
            print(f"[ERROR] Tidak dapat membuka kamera {selected_camera}")

# Thread untuk capture frame
capture_thread = threading.Thread(target=capture_frames, daemon=True)

def cleanup():
    """
    Fungsi untuk membersihkan resource saat aplikasi shutdown.
    """
    global cap
    with lock:
        if cap is not None:
            cap.release()
            cap = None
    print("[INFO] Aplikasi dimatikan, kamera dilepas.")

atexit.register(cleanup)

if __name__ == "__main__":
    initialize_camera()
    if not capture_thread.is_alive():
        capture_thread.start()
    print("[INFO] Server mulai berjalan...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
