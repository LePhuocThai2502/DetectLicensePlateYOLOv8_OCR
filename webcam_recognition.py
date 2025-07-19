import os
import cv2
import sqlite3
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template_string, send_from_directory, make_response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from paddleocr import PaddleOCR
from pyngrok import ngrok
from collections import Counter
import logging
import time
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Load YOLO model
try:
    model = YOLO("E:\Cameraonline\yolov8x_finetuned.pt")  # CẬP NHẬT ĐƯỜNG DẪN NÀY
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Error loading YOLO model: {str(e)}")
    raise

# Initialize SQLite database
def init_db():
    try:
        conn = sqlite3.connect("plates.db")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                plate_text TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

init_db()

# Utility function to safely delete files
def safe_delete_file(filepath):
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Deleted file: {filepath}")
        else:
            logger.warning(f"File not found for deletion: {filepath}")
    except Exception as e:
        logger.error(f"Error deleting file {filepath}: {str(e)}")

# Perspective correction for license plate
def correct_perspective(image, box):
    try:
        x1, y1, x2, y2 = map(int, box)
        margin = 10
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image.shape[1], x2 + margin)
        y2 = min(image.shape[0], y2 + margin)
        plate_img = image[y1:y2, x1:x2]
        if plate_img.size == 0:
            logger.warning("Empty plate image after cropping")
            return plate_img
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning("No contours found for perspective correction")
            return plate_img
        largest_contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        if len(approx) != 4:
            logger.warning("Could not find 4 corners for perspective correction")
            return plate_img
        corners = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]
        rect[2] = corners[np.argmax(s)]
        d = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(d)]
        rect[3] = corners[np.argmax(d)]
        width = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])))
        height = int(max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2])))
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(plate_img, M, (width, height))
        logger.debug("Perspective correction applied successfully")
        return warped
    except Exception as e:
        logger.error(f"Error in perspective correction: {str(e)}")
        return image[y1:y2, x1:x2]

# Preprocess image for OCR
def preprocess_image(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        resized = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return resized
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return image

# Initialize PaddleOCR
def get_ocr():
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang='en', cls=True, rec=True, det=True, use_gpu=False)
        logger.debug("PaddleOCR initialized")
        return ocr
    except Exception as e:
        logger.error(f"Error initializing PaddleOCR: {str(e)}")
        raise

# Process OCR
def process_ocr(image, boxes, ocr):
    plate_text = "Không phát hiện"
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        corrected_plate = correct_perspective(image, box)
        if corrected_plate.size == 0:
            logger.warning("Empty corrected plate, skipping")
            continue
        try:
            preprocessed_plate = preprocess_image(corrected_plate)
            ocr_result = ocr.ocr(preprocessed_plate, cls=True)
            logger.debug(f"OCR result: {ocr_result}")
            if ocr_result and ocr_result[0]:
                detected_texts = [item[1][0] for item in ocr_result[0] if item[1][0]]
                if detected_texts:
                    plate_text = " ".join(detected_texts).replace("\n", " ")
                    logger.info(f"Detected plate: {plate_text}")
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    break
        except Exception as e:
            logger.error(f"OCR processing error: {str(e)}")
    return plate_text, image

# HTML templates
HTML_STYLE = '''
<link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">
<style>
  body {
    font-family: 'Roboto', sans-serif;
    background: #f0f2f5;
    color: #333;
    margin: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  h2, h3 {
    color: #0d6efd;
    margin-top: 0;
  }
  h2::before {
    content: "📌 ";
  }
  form {
    background: #ffffff;
    padding: 20px;
    border-radius: 12px;
    max-width: 500px;
    width: 100%;
    margin: 0 auto 40px auto;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  }
  input[type=file], input[type=submit] {
    padding: 12px;
    width: 100%;
    margin-top: 10px;
    margin-bottom: 15px;
    border: 1px solid #ccc;
    border-radius: 6px;
    font-size: 16px;
    box-sizing: border-box;
  }
  input[type=submit] {
    background: #0d6efd;
    color: white;
    border: none;
    transition: 0.3s;
    cursor: pointer;
  }
  input[type=submit]:hover {
    background: #084298;
  }
  a {
    color: #0d6efd;
    text-decoration: none;
    margin: 10px;
  }
  a:hover {
    text-decoration: underline;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    max-width: 800px;
    background: white;
    box-shadow: 0 0 10px rgba(0,0,0,.05);
    margin-top: 30px;
  }
  th, td {
    border: 1px solid #ddd;
    padding: 10px;
    text-align: center;
  }
  th {
    background: #0d6efd;
    color: white;
  }
  img, video {
    max-width: 100%;
    height: auto;
    border: 1px solid #ccc;
    border-radius: 8px;
    box-shadow: 0 0 8px rgba(0,0,0,.1);
    margin: 20px auto;
    display: block;
  }
  .error {
    color: red;
    font-weight: bold;
  }
</style>
'''

HTML_INDEX = HTML_STYLE + '''
<h2>📷 Upload ảnh biển số</h2>
<form method="post" enctype="multipart/form-data" action="/">
  <input type="file" name="image" accept="image/*">
  <input type="submit" value="Nhận diện ảnh">
</form>
<a href="/video">📼 Nhận diện video</a>
<a href="/webcam">📷 Nhận diện qua webcam</a>
<a href="/history">📜 Lịch sử</a>
<a href="/export">📤 Xuất CSV</a>
'''

HTML_RESULT = HTML_STYLE + '''
<h2>Kết quả nhận diện ảnh</h2>
{% if error %}
<p class="error">{{ error }}</p>
{% else %}
<p><b>Biển số:</b> {{ plate }}</p>
<img src="{{ url_for('static', filename='uploads/' + filename) }}?t={{ timestamp }}" alt="Result Image">
{% endif %}
<br><a href="/">← Quay lại</a>
'''

HTML_VIDEO = HTML_STYLE + '''
<h2>📼 Upload video để nhận diện</h2>
<form method="post" enctype="multipart/form-data">
  <input type="file" name="video" accept="video/*">
  <input type="submit" value="Xử lý video">
</form>
<a href="/">← Quay lại</a>
'''

HTML_VIDEO_RESULT = HTML_STYLE + '''
<h2>✅ Xử lý video thành công!</h2>
<video controls>
    <source src="{{ url_for('static', filename='uploads/' + output_video) }}?t={{ timestamp }}" type="video/mp4">
    Trình duyệt của bạn không hỗ trợ video.
</video><br>
<a href="/">← Quay lại</a>
'''

HTML_WEBCAM = HTML_STYLE + '''
<h2>📷 Nhận diện qua Webcam</h2>
<p>Đã kích hoạt webcam. Kiểm tra cửa sổ OpenCV để xem luồng video.</p>
<p>Kết quả nhận diện sẽ được lưu vào lịch sử và có thể xem tại <a href="/history">Lịch sử</a>.</p>
<p>Nhấn 'q' trong cửa sổ webcam để thoát.</p>
<br><a href="/">← Quay lại</a>
'''

HTML_HISTORY = HTML_STYLE + '''
<h2>Lịch sử nhận diện</h2>
<table>
<tr><th>ID</th><th>Ảnh/Video</th><th>Biển số</th><th>Thời gian</th></tr>
{% for r in results %}
<tr><td>{{ r[0] }}</td><td><a href="{{ url_for('static', filename='uploads/' + r[1]) }}" target="_blank">{{ r[1] }}</a></td><td>{{ r[2] }}</td><td>{{ r[3] }}</td></tr>
{% endfor %}
</table>
<br><a href="/">← Trang chủ</a>
'''

# Utility functions
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Webcam recognition function
def webcam_recognition():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Không thể mở webcam")
            return False
        ocr = get_ocr()
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Không thể đọc frame từ webcam")
                break
            frame_count += 1
            if frame_count % 5 == 0:
                try:
                    results = model.predict(source=frame, conf=0.3, save=False)
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    logger.debug(f"YOLO detected {len(boxes)} boxes in frame {frame_count}")
                    plate_text, processed_frame = process_ocr(frame, boxes, ocr)
                    if plate_text != "Không phát hiện":
                        timestamp = str(int(time.time()))
                        result_filename = f"webcam_{timestamp}.jpg"
                        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                        cv2.imwrite(result_path, processed_frame)
                        conn = sqlite3.connect("plates.db")
                        conn.execute("INSERT INTO results (filename, plate_text, timestamp) VALUES (?, ?, ?)", (
                            result_filename, plate_text, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ))
                        conn.commit()
                        conn.close()
                        logger.info(f"Saved plate {plate_text} to database")
                    cv2.imshow("Webcam License Plate Recognition", processed_frame)
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {str(e)}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam recognition stopped")
        return True
    except Exception as e:
        logger.error(f"Error in webcam recognition: {str(e)}")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        return False

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'image' not in request.files:
            return render_template_string(HTML_RESULT, error="Không có file được chọn")
        file = request.files['image']
        if not file or file.filename == '':
            return render_template_string(HTML_RESULT, error="Không có file được chọn")
        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return render_template_string(HTML_RESULT, error="Định dạng file không hợp lệ. Chỉ chấp nhận PNG, JPG, JPEG")
        try:
            timestamp = str(int(time.time()))
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"image_{timestamp}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image = cv2.imread(filepath)
            if image is None:
                safe_delete_file(filepath)
                return render_template_string(HTML_RESULT, error="Không thể đọc ảnh")
            ocr = get_ocr()
            results = model.predict(source=filepath, conf=0.3, save=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            logger.debug(f"YOLO detected {len(boxes)} boxes")
            plate_text, processed_image = process_ocr(image, boxes, ocr)
            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            cv2.imwrite(result_path, processed_image)
            conn = sqlite3.connect("plates.db")
            conn.execute("INSERT INTO results (filename, plate_text, timestamp) VALUES (?, ?, ?)", (
                result_filename, plate_text, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
            conn.close()
            safe_delete_file(filepath)
            response = make_response(render_template_string(HTML_RESULT, filename=result_filename, plate=plate_text, timestamp=timestamp))
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            return response
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            safe_delete_file(filepath)
            return render_template_string(HTML_RESULT, error=f"Lỗi xử lý ảnh: {str(e)}")
    return render_template_string(HTML_INDEX)

@app.route("/video", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        if 'video' not in request.files:
            return render_template_string(HTML_VIDEO, error="Không có file được chọn")
        video = request.files['video']
        if not video or video.filename == '':
            return render_template_string(HTML_VIDEO, error="Không có file được chọn")
        if not allowed_file(video.filename, ALLOWED_VIDEO_EXTENSIONS):
            return render_template_string(HTML_VIDEO, error="Định dạng file không hợp lệ. Chỉ chấp nhận MP4, AVI, MOV")
        try:
            timestamp = str(int(time.time()))
            ext = video.filename.rsplit('.', 1)[1].lower()
            filename = f"video_{timestamp}.{ext}"
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"output_{filename}")
            video.save(input_path)
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                safe_delete_file(input_path)
                return render_template_string(HTML_VIDEO, error="Không thể mở video")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 25.0, (int(cap.get(3)), int(cap.get(4))))
            plate_buffer = []
            ocr = get_ocr()
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % 5 != 0:
                    out.write(frame)
                    continue
                try:
                    results = model.predict(source=frame, conf=0.3, save=False)
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    logger.debug(f"YOLO detected {len(boxes)} boxes in frame {frame_count}")
                    plate_text, processed_frame = process_ocr(frame, boxes, ocr)
                    if plate_text != "Không phát hiện":
                        plate_buffer.append(plate_text)
                    out.write(processed_frame)
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {str(e)}")
                    out.write(frame)
            cap.release()
            out.release()
            filtered = [p for p, c in Counter(plate_buffer).items() if c >= 2 and len(p) >= 5]
            conn = sqlite3.connect("plates.db")
            for p in filtered:
                conn.execute("INSERT INTO results (filename, plate_text, timestamp) VALUES (?, ?, ?)", (
                    f"output_{filename}", p, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
            conn.commit()
            conn.close()
            safe_delete_file(input_path)
            response = make_response(render_template_string(HTML_VIDEO_RESULT, output_video=f"output_{filename}", timestamp=timestamp))
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            return response
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            safe_delete_file(input_path)
            safe_delete_file(output_path)
            return render_template_string(HTML_VIDEO, error=f"Lỗi xử lý video: {str(e)}")
    return render_template_string(HTML_VIDEO)

@app.route("/webcam", methods=["GET"])
def webcam():
    success = webcam_recognition()
    if not success:
        return render_template_string(HTML_STYLE + '''
            <h2>📷 Nhận diện qua Webcam</h2>
            <p class="error">Lỗi: Không thể mở webcam. Kiểm tra thiết bị và thử lại.</p>
            <br><a href="/">← Quay lại</a>
        ''')
    return render_template_string(HTML_WEBCAM)

@app.route("/history")
def history():
    try:
        conn = sqlite3.connect("plates.db")
        results = conn.execute("SELECT * FROM results ORDER BY timestamp DESC").fetchall()
        conn.close()
        return render_template_string(HTML_HISTORY, results=results)
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        return render_template_string(HTML_STYLE + '<h2>Lịch sử nhận diện</h2><p class="error">Lỗi: Không thể tải lịch sử</p><br><a href="/">← Quay lại</a>')

@app.route("/export")
def export_csv():
    csv_path = os.path.join("static", "plates_export.csv")
    try:
        conn = sqlite3.connect("plates.db")
        df = pd.read_sql_query("SELECT * FROM results", conn)
        df.to_csv(csv_path, index=False)
        conn.close()
        response = make_response(render_template_string(HTML_STYLE + '''
            <h2>✅ Xuất CSV thành công!</h2>
            <a href="/static/plates_export.csv" target="_blank">Tải CSV</a><br><br>
            <a href="/">← Quay lại</a>
        '''))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response
    except Exception as e:
        logger.error(f"Error exporting CSV: {str(e)}")
        safe_delete_file(csv_path)
        return render_template_string(HTML_STYLE + '<h2>Xuất CSV</h2><p class="error">Lỗi: Không thể xuất CSV</p><br><a href="/">← Quay lại</a>')

@app.route('/static/<path:filename>')
def serve_static(filename):
    try:
        response = send_from_directory('static', filename)
        if filename == "plates_export.csv":
            safe_delete_file(os.path.join('static', filename))
        return response
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        return render_template_string(HTML_STYLE + f'<h2>Lỗi</h2><p class="error">Không thể tải file: {str(e)}</p><br><a href="/">← Quay lại</a>')

# Set ngrok auth token and run app
NGROK_AUTHTOKEN = "2xDRyGFZ8gy0nQHQfU8CXM4agaJ_4KMCshjcvtc5CatExXpgM"  # THAY BẰNG TOKEN CỦA BẠN
ngrok.set_auth_token(NGROK_AUTHTOKEN)
public_url = ngrok.connect(5000)
print("🔗 Truy cập web tại:", public_url)
app.run(host='0.0.0.0', port=5000)