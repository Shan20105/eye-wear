from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import uuid
from keras.models import load_model
import dlib
from PIL import Image

app = Flask(__name__)

# Folders
UPLOAD_FOLDER = 'static/uploaded'
OUTPUT_FOLDER = 'static/output'
EYEWEAR_FOLDER = 'static/eyewear'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load model & detectors
model = load_model('faceshape_model.h5')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

# Face shape labels
labels = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

# Resize image for display
def resize_image(image_path, output_path=None, max_size=(500, 500)):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    scale = min(max_size[0] / w, max_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = 255 * np.ones((max_size[1], max_size[0], 3), dtype=np.uint8)
    start_x = (max_size[0] - new_w) // 2
    start_y = (max_size[1] - new_h) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized

    save_path = output_path if output_path else image_path
    cv2.imwrite(save_path, canvas)
    return save_path

# Overlay eyewear on face
def overlay_eyewear_on_face(user_img_path, eyewear_path, face_rect, output_path):
    user_img = Image.open(user_img_path).convert("RGBA")
    eyewear_img = Image.open(eyewear_path).convert("RGBA")

    left, top, right, bottom = face_rect
    face_width = right - left
    scale = face_width / eyewear_img.width
    new_eyewear_height = int(eyewear_img.height * scale)
    eyewear_resized = eyewear_img.resize((face_width, new_eyewear_height), Image.Resampling.LANCZOS)

    x_offset = left
    y_offset = top + int((bottom - top) * 0.25)

    user_img.paste(eyewear_resized, (x_offset, y_offset), eyewear_resized)
    user_img.save(output_path)
    return output_path

# Predict face shape
def detect_face_shape(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if not faces:
        return None, None

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (32, 32)).reshape(1, 32, 32, 1) / 255.0
        pred = model.predict(face_img)[0]
        label = labels[np.argmax(pred)]
        return label, (face.left(), face.top(), face.right(), face.bottom())
    return None, None

# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img_id = uuid.uuid4().hex
            img_path = os.path.join(UPLOAD_FOLDER, f"{img_id}.png")
            file.save(img_path)

            face_shape, face_rect = detect_face_shape(img_path)
            if not face_shape:
                return render_template('index.html', error="No face detected. Try another image.")

            eyewear_options = os.listdir(os.path.join(EYEWEAR_FOLDER, face_shape))
            eyewear_options = [f"{face_shape}/{ew}" for ew in eyewear_options]

            return render_template('select_eyewear.html',
                                   face_shape=face_shape,
                                   img_path=img_path,
                                   options=eyewear_options,
                                   face_rect=str(face_rect))

    return render_template('index.html')

# Apply eyewear
# @app.route('/apply', methods=['POST'])
# def apply_eyewear():
#     face_rect = eval(request.form['face_rect'])  # (left, top, right, bottom)
#     img_path = request.form['img_path']
#     eyewear_file = request.form['eyewear']

#     output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(img_path))
#     eyewear_path = os.path.join(EYEWEAR_FOLDER, eyewear_file)

#     print("Face Rect:", face_rect)
#     print("Image Path:", img_path)
#     print("Eyewear Path:", eyewear_path)

#     overlay_eyewear_on_face(img_path, eyewear_path, face_rect, output_path)

#     return render_template("final_result.html", final_image=output_path)
@app.route('/apply', methods=['POST'])
def apply_eyewear():
    print("Form Data:", request.form)
    face_rect = request.form['face_rect']
    img_path = request.form['img_path']
    eyewear_file = request.form['eyewear']
    output_filename = f"{uuid.uuid4().hex}.png"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename).replace('\\', '/')  # Fix slashes
    eyewear_path = os.path.join(EYEWEAR_FOLDER, eyewear_file).replace('\\', '/')  # Fix slashes
    print("Input Image Exists:", os.path.exists(img_path))
    print("Eyewear Image Exists:", os.path.exists(eyewear_path))
    print("Output Path:", output_path)
    try:
        overlay_eyewear_on_face(img_path, eyewear_path, eval(face_rect), output_path)
        print("Image Saved Successfully at:", output_path)
        if not os.path.exists(output_path):
            return "Error: Output image not saved!"
        return render_template("final_result.html", final_image=output_path)
    except Exception as e:
        print("Error in /apply:", str(e))
        return f"Error: {str(e)}"
# Run app
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(debug=True)