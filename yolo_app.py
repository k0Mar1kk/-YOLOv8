from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import json
from datetime import datetime
import time
import pandas as pd
from yolo_model import predict_image, predict_video

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = "results"
HISTORY_FILE = "request_history.json"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def save_to_history(data):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    
    history.append(data)
    
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def generate_excel_report(history_data, filename):
    try:
        df = pd.DataFrame(history_data)
        
        df = df[['timestamp', 'filename', 'objects_detected', 'processing_time', 'result_file']]
        
        df.columns = [
            'Дата и время', 
            'Имя файла', 
            'Обнаружено объектов', 
            'Время обработки (сек)', 
            'Файл результата'
        ]
        
        filepath = os.path.join(RESULT_FOLDER, filename)
        df.to_excel(filepath, index=False, engine='openpyxl')
        return True, filename
        
    except Exception as e:
        return False, str(e)

@app.route("/", methods=["GET", "POST"])
def index():
    filename = None
    stats = None
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            start_time = time.time()
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            if file.filename.endswith(('.mp4', '.avi', '.mov')):
                result_file, objects_count = predict_video(filepath)
            else:
                result_file, objects_count = predict_image(filepath)

            processing_time = round(time.time() - start_time, 2)
            filename = os.path.basename(result_file)
            
            stats = {
                'filename': file.filename,
                'objects_detected': objects_count,
                'processing_time': processing_time,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'result_file': filename
            }
            
            save_to_history(stats)
    return render_template('index.html', filename=filename, stats=stats)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/results/<path:filename>')
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/history')
def history():
    if not os.path.exists(HISTORY_FILE):
        return render_template('history.html', history=[])
    
    with open(HISTORY_FILE, 'r') as f:
        history_data = json.load(f)
    
    return render_template('history.html', history=history_data)

@app.route('/generate-report')
def generate_report():
    if not os.path.exists(HISTORY_FILE):
        return redirect(url_for('history'))
    
    with open(HISTORY_FILE, 'r') as f:
        history_data = json.load(f)
    report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    success, result = generate_excel_report(history_data, report_filename)
    if success:
        return send_from_directory(RESULT_FOLDER, report_filename, as_attachment=True)
    else:
        return f"Ошибка при генерации отчета: {result}", 500

if __name__ == '__main__':
    app.run(debug=True)