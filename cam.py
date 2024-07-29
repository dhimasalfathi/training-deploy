import torch
from ultralytics import YOLO
import cv2  # For image processing and frame capture
from flask import Flask, Response, render_template  # For web framework (Flask)
import numpy as np  # For image encoding
import socketIO  # For WebSocket communication

model = YOLO("newsafety.pt")
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML page

def gen_frames():
    # ... webcam capture, inference, and frame processing logic goes here ...
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')  # Stream video frames

if __name__ == '__main__':
    sio = socketIO.Server(async_mode='socketio.asyncio')  # Create WebSocket server
    app.wsgi_app = sio.wsgi_app

    @sio.event
    def connect(sid, environ):
        print('Client connected')

    @sio.event
    def receive_frame(sid, data):
        # ... receive encoded frame data, process it, perform inference, send predictions ...

    sio.run(app, host='0.0.0.0', port=5000)  # Run the server on any IP (0.0.0.0) and port 5000
