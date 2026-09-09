import argparse
import base64
import csv
import os
import socket
import time
import uuid
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from main import count_plate, imread_unicode, imwrite_unicode

ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(ROOT, "uploads")
DATA_DIR = os.path.join(ROOT, "data")
CSV_PATH = os.path.join(ROOT, "colony_counts.csv")
UPLOAD_TTL_SEC = 24 * 3600
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CSV_FIELDS = [
    "filename",
    "path",
    "mode",
    "full_red",
    "full_green",
    "full_blue",
    "full_total",
    "roi_red",
    "roi_green",
    "roi_blue",
    "roi_total",
    "roi_type",
    "roi_detail",
    "time",
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB batch uploads
JOBS = {}
PORT = 7860


def cleanup_old_uploads(ttl_sec=UPLOAD_TTL_SEC):
    """删除 uploads/data 中超过 ttl 的用户上传文件。"""
    now = time.time()
    removed = 0
    for folder in (UPLOAD_DIR, DATA_DIR):
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            try:
                mtime = os.path.getmtime(path)
                if now - mtime > ttl_sec:
                    os.remove(path)
                    removed += 1
            except OSError:
                continue
    return removed


def get_lan_ips():
    ips = []
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127.") and ":" not in ip:
                ips.append(ip)
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        if ip and not ip.startswith("127.") and ip not in ips:
            ips.insert(0, ip)
        s.close()
    except OSError:
        pass
    # de-dup keep order
    seen = set()
    out = []
    for ip in ips:
        if ip not in seen:
            seen.add(ip)
            out.append(ip)
    return out


def encode_jpeg(img, quality=85):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def in_roi(colony, roi):
    cx, cy = colony["cx"], colony["cy"]
    if roi["type"] == "circle":
        ox, oy, r = roi["cx"], roi["cy"], roi["r"]
        return (cx - ox) ** 2 + (cy - oy) ** 2 <= r * r
    pts = np.array(roi["points"], dtype=np.int32)
    if len(pts) < 3:
        return False
    return cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0


def roi_stats(colonies, roi):
    red = green = blue = 0
    for c in colonies:
        if not in_roi(c, roi):
            continue
        if c["kind"] == "big":
            red += 1
        elif c["kind"] == "small":
            green += 1
        else:
            blue += 1
    return {"red": red, "green": green, "blue": blue, "total": red + green + blue}


def render_roi(vis, colonies, roi, stats):
    out = vis.copy()
    overlay = out.copy()
    if roi["type"] == "circle":
        cx, cy, r = int(roi["cx"]), int(roi["cy"]), int(roi["r"])
        cv2.circle(overlay, (cx, cy), r, (0, 80, 80), -1)
        cv2.circle(out, (cx, cy), r, (0, 255, 255), 3)
        cv2.circle(out, (cx, cy), 6, (0, 255, 255), -1)
    else:
        pts = np.array(roi["points"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], (0, 80, 80))
        cv2.polylines(out, [pts], True, (0, 255, 255), 3)
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    for c in colonies:
        if not in_roi(c, roi):
            continue
        x, y, w, h = c["box"]
        color = (0, 0, 255) if c["kind"] == "big" else (0, 255, 0) if c["kind"] == "small" else (255, 0, 0)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
    cv2.putText(out, f"ROI red: {stats['red']}", (100, 900), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
    cv2.putText(out, f"ROI green: {stats['green']}", (100, 1050), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)
    cv2.putText(out, f"ROI blue: {stats['blue']}", (100, 1200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 6)
    cv2.putText(out, f"ROI total: {stats['total']}", (100, 1350), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 6)
    return out


def append_csv(row):
    write_header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def process_upload(f):
    name = f.filename or "image.jpg"
    base = os.path.basename(name.replace("\\", "/"))
    ext = os.path.splitext(base)[1].lower() or ".jpg"
    if ext not in IMAGE_EXTS:
        raise ValueError(f"不支持的格式: {base}")
    job_id = uuid.uuid4().hex
    raw_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    f.save(raw_path)
    img = imread_unicode(raw_path)
    if img is None:
        raise ValueError(f"无法读取: {base}")
    vis, colonies, counts, circle_info = count_plate(img)
    vis_path = os.path.join(UPLOAD_DIR, f"{job_id}_vis.jpg")
    imwrite_unicode(vis_path, vis)
    JOBS[job_id] = {
        "filename": base,
        "path": raw_path,
        "vis_path": vis_path,
        "vis": vis,
        "colonies": colonies,
        "counts": counts,
        "circle": {"cx": circle_info[0], "cy": circle_info[1], "r": circle_info[2]},
        "shape": {"h": vis.shape[0], "w": vis.shape[1]},
        "image": encode_jpeg(vis),
    }
    return {
        "job_id": job_id,
        "filename": base,
        "counts": {
            "red": counts["big"],
            "green": counts["small"],
            "blue": counts["tiny"],
            "total": counts["summary"],
        },
        "circle": JOBS[job_id]["circle"],
        "shape": JOBS[job_id]["shape"],
        "image": JOBS[job_id]["image"],
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/hostinfo")
def api_hostinfo():
    ips = get_lan_ips()
    return jsonify(
        {
            "port": PORT,
            "lan_ips": ips,
            "urls": [f"http://{ip}:{PORT}" for ip in ips] + [f"http://127.0.0.1:{PORT}"],
        }
    )


@app.post("/api/count")
def api_count():
    files = request.files.getlist("files")
    if not files:
        one = request.files.get("file")
        files = [one] if one and one.filename else []
    files = [f for f in files if f and f.filename]
    # filter non-images from directory picks
    picked = []
    for f in files:
        ext = os.path.splitext(os.path.basename(f.filename.replace("\\", "/")))[1].lower()
        if ext in IMAGE_EXTS:
            picked.append(f)
    if not picked:
        return jsonify({"error": "请上传图片或包含图片的目录"}), 400

    jobs = []
    failed = []
    for i, f in enumerate(picked, 1):
        try:
            jobs.append(process_upload(f))
        except Exception as e:
            failed.append({"filename": f.filename, "error": str(e)})
    if not jobs:
        return jsonify({"error": "全部失败", "failed": failed}), 400
    return jsonify({"jobs": jobs, "failed": failed, "total": len(jobs)})


@app.post("/api/roi")
def api_roi():
    data = request.get_json(force=True, silent=True) or {}
    job_id = data.get("job_id")
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "任务不存在，请重新上传"}), 404
    mode = data.get("mode", "circle")
    if mode == "circle":
        r = int(data.get("r", 0))
        if r <= 0:
            return jsonify({"error": "半径无效"}), 400
        roi = {"type": "circle", "cx": job["circle"]["cx"], "cy": job["circle"]["cy"], "r": r}
        roi_detail = f"cx={roi['cx']};cy={roi['cy']};r={roi['r']}"
    else:
        points = data.get("points") or []
        if len(points) < 3:
            return jsonify({"error": "曲线点数不足"}), 400
        pts = [[int(p[0]), int(p[1])] for p in points]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        roi = {"type": "curve", "points": pts}
        roi_detail = f"points={len(pts)}"
    stats = roi_stats(job["colonies"], roi)
    out = render_roi(job["vis"], job["colonies"], roi, stats)
    c = job["counts"]
    row = {
        "filename": job["filename"],
        "path": os.path.abspath(job["path"]),
        "mode": mode,
        "full_red": c["big"],
        "full_green": c["small"],
        "full_blue": c["tiny"],
        "full_total": c["summary"],
        "roi_red": stats["red"],
        "roi_green": stats["green"],
        "roi_blue": stats["blue"],
        "roi_total": stats["total"],
        "roi_type": roi["type"],
        "roi_detail": roi_detail,
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    append_csv(row)
    b64 = encode_jpeg(out)
    job["image"] = b64
    return jsonify({"roi": stats, "image": b64, "csv": CSV_PATH, "row": row})


@app.get("/api/csv")
def api_csv():
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
    return send_file(CSV_PATH, as_attachment=True, download_name="colony_counts.csv")


def main():
    global PORT
    parser = argparse.ArgumentParser()
    parser.add_argument("port", nargs="?", type=int, default=7860)
    args = parser.parse_args()
    PORT = args.port
    n = cleanup_old_uploads()
    ips = get_lan_ips()
    print("=" * 50)
    print("菌落计数 Web 服务已启动")
    print(f"已清理 24h 前上传文件: {n} 个")
    print(f"本机访问:  http://127.0.0.1:{PORT}")
    if ips:
        print("局域网访问:")
        for ip in ips:
            print(f"  http://{ip}:{PORT}")
    else:
        print("未探测到局域网 IP，请检查网卡")
    print("=" * 50)
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
