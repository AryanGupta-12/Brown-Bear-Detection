from flask import Flask, render_template, request, redirect, url_for, Response, send_from_directory
import os
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['PROCESSED_FOLDER']='static/uploads/processed/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
def clear_upload_folders():
    for file in [app.config['UPLOAD_FOLDER'],app.config["PROCESSED_FOLDER"]]:
        if os.path.exists(file):
            shutil.rmtree(file)
        os.makedirs(file)

clear_upload_folders()

video_cap = None
model = None
processing_stopped = False
output_video_path = ''

def process_image(image_path):
    image = cv2.imread(image_path)
    
    model = YOLO(r'best.pt')

    results = model(image)
    
    annotator = Annotator(image)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
    image = annotator.result()  
    
    output_path = image_path.replace('uploads', 'uploads/processed')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    
    return output_path
    
def process_video(video_path):
    global video_cap
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print("Error opening video stream or file")
        video_cap= None
        return None

    return video_path.replace('static/', '')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global video_cap, model, processing_stopped

    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        if file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
            output_path = process_image(file_path)
        else:
            output_path = process_video(file_path)
            output_path = output_path.replace('static/', '')
            return redirect(url_for('video_feed', file_path=output_path))
        output_path = output_path.replace('static', '')
        return render_template('index.html', file_path=output_path)
    return redirect(request.url)

@app.route('/stop_processing')
def stop_processing():
    global processing_stopped
    processing_stopped = True
    return redirect(url_for('index'))

@app.route('/download/<path:filename>')
def download_file(filename):
    print(filename)
    return send_from_directory('static/', filename, as_attachment=True)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global video_cap, model

    model = YOLO(r'best.pt')


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(app.config['PROCESSED_FOLDER'], 'output.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while video_cap.isOpened() and not processing_stopped:
        ret, frame = video_cap.read()
        if not ret:
            break

        results = model(frame)
        
        annotator = Annotator(frame)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  
                c = box.cls
                annotator.box_label(b, model.names[int(c)])
        frame = annotator.result()
        out.write(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    video_cap.release()
    out.release()

if __name__ == '__main__':
    app.run(debug=True)
