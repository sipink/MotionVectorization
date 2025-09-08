import os, uuid, json
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():
    # Phase 1: DRY-RUN (no GPU yet)
    video_url = request.form.get("video_url", "").strip()
    start_frame = int(request.form.get("start_frame", "0"))
    max_frames = int(request.form.get("max_frames", "120"))
    notes = request.form.get("notes", "").strip()

    # Accept file OR URL (file handling is Phase 2 via object storage / network volume)
    file = request.files.get("video_file")
    file_info = None
    if file and file.filename:
        # Placeholder only for Phase 1
        file_info = {"filename": file.filename, "size": request.content_length}

    payload = {
        "git_ref": os.getenv("GIT_REF", "main"),
        "video_url": video_url if video_url else None,
        "file_info": file_info,
        "config": {
            "start_frame": start_frame,
            "max_frames": max_frames
        },
        "notes": notes
    }

    job_id = f"dryrun-{uuid.uuid4().hex[:8]}"
    return jsonify({"status": "queued (dry-run)", "job_id": job_id, "would_send": payload})

if __name__ == "__main__":
    # Replit favors port 8000
    app.run(host="0.0.0.0", port=8000, debug=True)