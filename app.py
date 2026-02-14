import os
import cv2
from flask import Flask, request, jsonify, send_file
from processing.pipeline import EnhancementPipeline
from metrics.psnr import calculate_psnr
from metrics.entropy import calculate_entropy

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route("/enhance", methods=["POST"])
def enhance_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    gamma = float(request.form.get("gamma", 1.2))
    clip_limit = float(request.form.get("clip_limit", 3.0))

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    image = cv2.imread(filepath)

    pipeline = EnhancementPipeline(gamma=gamma, clip_limit=clip_limit)
    enhanced = pipeline.process(image)

    output_path = os.path.join(PROCESSED_FOLDER, "enhanced_" + file.filename)
    cv2.imwrite(output_path, enhanced)

    psnr = calculate_psnr(image, enhanced)
    entropy = calculate_entropy(enhanced)

    return jsonify({
        "message": "Image processed successfully",
        "psnr": round(float(psnr), 2),
        "entropy": round(float(entropy), 2),
        "download_url": f"/download/{'enhanced_' + file.filename}"
    })

@app.route("/download/<filename>")
def download_file(filename):
    path = os.path.join(PROCESSED_FOLDER, filename)
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
