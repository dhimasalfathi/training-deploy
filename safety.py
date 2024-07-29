from flask import (
    Flask,
    request,
    send_file,
    render_template,
    url_for,
    flash,
    redirect,
)
import cv2
from ultralytics import YOLO
import os
import glob

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for flashing messages


# Function to resize the video
def resize_video(input_path, output_path, max_height=480):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if height > max_height:
        scale = max_height / height
        new_width = int(width * scale)
        new_height = max_height
    else:
        new_width = width
        new_height = height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (new_width, new_height))
        out.write(resized_frame)

    cap.release()
    out.release()


# Load YOLO model
model = YOLO("newsafety.pt")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            if file.filename == "":
                flash("No selected file")
                return redirect(request.url)

            # Save the uploaded file
            input_path = os.path.join("uploads", file.filename)
            file.save(input_path)

            # Resize the video
            resized_video_path = os.path.join("uploads", "resized_" + file.filename)
            resize_video(input_path, resized_video_path)

            # Run YOLO inference on uploaded video
            results = model(
                source=resized_video_path,
                show=False,
                conf=0.6,
                save=True,
                classes=[
                    5,
                    6,
                    7,
                    9,
                    12,
                    13,
                    14,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                ],
            )

            # Find the latest output file path
            output_folder = os.path.join("runs", "detect")
            latest_run = max(
                glob.glob(os.path.join(output_folder, "predict*")), key=os.path.getmtime
            )
            output_files = glob.glob(os.path.join(latest_run, "*.avi"))

            if not output_files:
                flash("Processed video not found. Something went wrong.")
                return redirect("/")

            output_video_path = output_files[0]

            return render_template(
                "index.html",
                video_url=url_for(
                    "uploads", filename=os.path.basename(output_video_path)
                ),
            )

        elif "use_webcam" in request.form:
            # Run YOLO inference on webcam
            results = model(
                source="0",
                show=True,
                conf=0.4,
                save=True,
                classes=[
                    5,
                    6,
                    7,
                    9,
                    12,
                    13,
                    14,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                ],
            )

            # Find the latest output file path
            output_folder = os.path.join("runs", "detect")
            latest_run = max(
                glob.glob(os.path.join(output_folder, "predict*")), key=os.path.getmtime
            )
            output_files = glob.glob(os.path.join(latest_run, "*.avi"))

            if not output_files:
                flash("Processed video not found. Something went wrong.")
                return redirect("/")

            output_video_path = output_files[0]

            return render_template(
                "index.html",
                video_url=url_for(
                    "uploads", filename=os.path.basename(output_video_path)
                ),
            )

    return render_template("index.html", video_url=None)


@app.route("/uploads/<filename>")
def uploads(filename):
    output_folder = os.path.join("runs", "detect")
    latest_run = max(
        glob.glob(os.path.join(output_folder, "predict*")), key=os.path.getmtime
    )
    return send_file(os.path.join(latest_run, filename))


if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
