import os
import uuid
import cv2
from flask import Flask, request, jsonify, send_file, render_template, url_for
from flask_cors import CORS
from processing.pipeline import EnhancementPipeline
from metrics.psnr import calculate_psnr
from metrics.entropy import calculate_entropy

app = Flask(__name__)
CORS(app)

# ===============================
# Configuration
# ===============================
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB limit


# ===============================
# Homepage Route (Frontend)
# ===============================
@app.route("/")
def home():
    return render_template("index.html")


# ===============================
# Image Enhancement Route
# ===============================
@app.route("/enhance", methods=["POST"])
def enhance_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Unique filename (avoid overwrite issues)
        unique_id = str(uuid.uuid4())
        filename = unique_id + "_" + file.filename

        gamma = float(request.form.get("gamma", 1.2))
        clip_limit = float(request.form.get("clip_limit", 3.0))

        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(upload_path)

        # Read Image
        image = cv2.imread(upload_path)

        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Process Image
        pipeline = EnhancementPipeline(gamma=gamma, clip_limit=clip_limit)
        enhanced = pipeline.process(image)

        output_filename = "enhanced_" + filename
        output_path = os.path.join(app.config["PROCESSED_FOLDER"], output_filename)
        cv2.imwrite(output_path, enhanced)

        # Metrics
        psnr = calculate_psnr(image, enhanced)
        entropy = calculate_entropy(enhanced)

        return jsonify({
            "message": "Image processed successfully",
            "psnr": round(float(psnr), 2),
            "entropy": round(float(entropy), 2),
            "original_image_url": url_for("uploaded_file", filename=filename),
            "enhanced_image_url": url_for("processed_file", filename=output_filename),
            "download_url": url_for("download_file", filename=output_filename)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===============================
# Serve Uploaded Image
# ===============================
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    return send_file(path)


# ===============================
# Serve Processed Image
# ===============================
@app.route("/processed/<filename>")
def processed_file(filename):
    path = os.path.join(app.config["PROCESSED_FOLDER"], filename)
    return send_file(path)


# ===============================
# Download Route
# ===============================
@app.route("/download/<filename>")
def download_file(filename):
    path = os.path.join(app.config["PROCESSED_FOLDER"], filename)
    return send_file(path, as_attachment=True)


# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
