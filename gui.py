import argparse
import csv
import glob
import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from main import count_plate, imread_unicode


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
]

class PlateCounterGUI:
    def __init__(self, paths):
        self.paths = paths
        self.idx = 0
        self.vis = None
        self.colonies = []
        self.counts = None
        self.circle_info = None
        self.scale = 1.0
        self.drawing = False
        self.canvas_pts = []
        self.roi_pts = None
        self.roi_circle = None
        self.preview_radius = None
        self.tk_img = None

        self.root = tk.Tk()
        self.root.title("菌落计数")
        self.root.geometry("1280x900")

        bar = tk.Frame(self.root)
        bar.pack(fill=tk.X)
        tk.Button(bar, text="打开", command=self.open_files).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(bar, text="上一张", command=self.prev_image).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="下一张", command=self.next_image).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="清除选区", command=self.clear_roi).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="导出CSV", command=self.export_csv).pack(side=tk.LEFT, padx=4)

        self.mode = tk.StringVar(value="circle")
        mode_box = tk.LabelFrame(bar, text="选取方式", padx=4, pady=0)
        mode_box.pack(side=tk.LEFT, padx=8)
        tk.Radiobutton(
            mode_box, text="自由曲线", variable=self.mode, value="curve", command=self.on_mode_change
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            mode_box, text="霍夫圆心拉圆", variable=self.mode, value="circle", command=self.on_mode_change
        ).pack(side=tk.LEFT)

        self.csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colony_counts.csv")
        self.auto_csv = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="松手自动写入CSV", variable=self.auto_csv).pack(side=tk.LEFT, padx=4)

        self.info = tk.Label(bar, text="", anchor="w")
        self.info.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.set_hint()

        self.canvas = tk.Canvas(self.root, bg="#222", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind("<Configure>", self.on_resize)
        self._last_canvas = (0, 0)

        if self.paths:
            self.root.after(80, self.load_current)

    def set_hint(self):
        if self.mode.get() == "circle":
            tip = "拉圆：以霍夫圆圆心为中心，拖动鼠标定半径；选区内分别统计 红/绿/蓝 框"
        else:
            tip = "自由曲线：按住左键拖出闭合曲线；选区内分别统计 红/绿/蓝 框"
        self.info.config(text=tip)

    def on_mode_change(self):
        self.clear_roi()

    def on_resize(self, event):
        if event.widget is not self.root or self.vis is None:
            return
        size = (self.canvas.winfo_width(), self.canvas.winfo_height())
        if size == self._last_canvas or size[0] < 50:
            return
        self._last_canvas = size
        self.refresh()

    def open_files(self):
        files = filedialog.askopenfilenames(
            title="选择平板图片",
            filetypes=[("图片", "*.jpg;*.jpeg;*.png;*.JPG;*.JPEG;*.PNG"), ("All", "*.*")],
        )
        if not files:
            return
        self.paths = list(files)
        self.idx = 0
        self.load_current()

    def prev_image(self):
        if not self.paths:
            return
        self.idx = (self.idx - 1) % len(self.paths)
        self.load_current()

    def next_image(self):
        if not self.paths:
            return
        self.idx = (self.idx + 1) % len(self.paths)
        self.load_current()

    def load_current(self):
        path = self.paths[self.idx]
        img = imread_unicode(path)
        if img is None:
            messagebox.showerror("错误", f"无法读取 {path}")
            return
        self.root.config(cursor="watch")
        self.root.update()
        try:
            self.vis, self.colonies, self.counts, self.circle_info = count_plate(img)
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("计数失败", str(e))
            return
        self.root.config(cursor="")
        self.clear_roi(refresh=False)
        self.refresh()

    def canvas_to_img(self, x, y):
        return int(x / self.scale), int(y / self.scale)

    def img_to_canvas(self, x, y):
        return x * self.scale, y * self.scale

    def has_roi(self):
        if self.roi_circle is not None:
            return True
        return self.roi_pts is not None and len(self.roi_pts) >= 3

    def in_roi(self, cx, cy):
        if self.roi_circle is not None:
            ox, oy, r = self.roi_circle
            return (cx - ox) ** 2 + (cy - oy) ** 2 <= r * r
        if self.roi_pts is None or len(self.roi_pts) < 3:
            return False
        return cv2.pointPolygonTest(self.roi_pts, (float(cx), float(cy)), False) >= 0

    def roi_stats(self):
        if not self.has_roi():
            return None
        red = green = blue = 0
        for c in self.colonies:
            if not self.in_roi(c["cx"], c["cy"]):
                continue
            if c["kind"] == "big":
                red += 1
            elif c["kind"] == "small":
                green += 1
            else:
                blue += 1
        return {"red": red, "green": green, "blue": blue, "total": red + green + blue}

    def draw_preview(self):
        self.canvas.delete("preview")
        if self.mode.get() == "circle":
            if self.circle_info is None or self.preview_radius is None:
                return
            cx, cy, _ = self.circle_info
            sx, sy = self.img_to_canvas(cx, cy)
            sr = self.preview_radius * self.scale
            self.canvas.create_oval(
                sx - sr, sy - sr, sx + sr, sy + sr,
                outline="#ffff00", width=2, tags="preview",
            )
            self.canvas.create_oval(
                sx - 4, sy - 4, sx + 4, sy + 4,
                fill="#ffff00", outline="", tags="preview",
            )
            return
        if len(self.canvas_pts) < 2:
            return
        flat = [c for p in self.canvas_pts for c in p]
        self.canvas.create_line(*flat, fill="#ffff00", width=2, tags="preview", smooth=True)
        if len(self.canvas_pts) >= 3:
            x0, y0 = self.canvas_pts[0]
            x1, y1 = self.canvas_pts[-1]
            self.canvas.create_line(x1, y1, x0, y0, fill="#ffff00", width=2, dash=(4, 3), tags="preview")

    def render(self):
        vis = self.vis.copy()
        if self.has_roi():
            overlay = vis.copy()
            if self.roi_circle is not None:
                cx, cy, r = self.roi_circle
                cv2.circle(overlay, (cx, cy), r, (0, 80, 80), -1)
                cv2.circle(vis, (cx, cy), r, (0, 255, 255), 3)
                cv2.circle(vis, (cx, cy), 6, (0, 255, 255), -1)
            else:
                pts = self.roi_pts.reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [pts], (0, 80, 80))
                cv2.polylines(vis, [pts], True, (0, 255, 255), 3)
            vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
            for c in self.colonies:
                if not self.in_roi(c["cx"], c["cy"]):
                    continue
                x, y, w, h = c["box"]
                if c["kind"] == "big":
                    color = (0, 0, 255)
                elif c["kind"] == "small":
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3)
            st = self.roi_stats()
            cv2.putText(vis, f"ROI red: {st['red']}", (100, 900), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
            cv2.putText(vis, f"ROI green: {st['green']}", (100, 1050), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)
            cv2.putText(vis, f"ROI blue: {st['blue']}", (100, 1200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 6)
            cv2.putText(vis, f"ROI total: {st['total']}", (100, 1350), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 6)

        h, w = vis.shape[:2]
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        max_w = cw if cw > 50 else 1200
        max_h = ch if ch > 50 else 800
        self.scale = min(max_w / w, max_h / h, 1.0)
        dw, dh = int(w * self.scale), int(h * self.scale)
        rgb = cv2.cvtColor(cv2.resize(vis, (dw, dh)), cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb)), dw, dh

    def refresh(self):
        if self.vis is None:
            return
        self.tk_img, dw, dh = self.render()
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        name = os.path.basename(self.paths[self.idx])
        c = self.counts
        text = (
            f"{name}  ({self.idx + 1}/{len(self.paths)})  "
            f"全图 红={c['big']} 绿={c['small']} 蓝={c['tiny']} 合计={c['summary']}"
        )
        st = self.roi_stats()
        if st is not None:
            text += (
                f"  |  选区 红={st['red']} 绿={st['green']} 蓝={st['blue']} 合计={st['total']}"
            )
        self.info.config(text=text)

    def on_press(self, event):
        if self.vis is None:
            return
        self.drawing = True
        self.roi_pts = None
        self.roi_circle = None
        self.preview_radius = None
        self.canvas.delete("preview")
        if self.mode.get() == "circle":
            if self.circle_info is None:
                messagebox.showwarning("提示", "当前图未检测到霍夫圆心")
                self.drawing = False
                return
            mx, my = self.canvas_to_img(event.x, event.y)
            cx, cy, _ = self.circle_info
            self.preview_radius = max(1, int(math.hypot(mx - cx, my - cy)))
            self.draw_preview()
            return
        self.canvas_pts = [(event.x, event.y)]

    def on_drag(self, event):
        if not self.drawing:
            return
        if self.mode.get() == "circle":
            mx, my = self.canvas_to_img(event.x, event.y)
            cx, cy, _ = self.circle_info
            self.preview_radius = max(1, int(math.hypot(mx - cx, my - cy)))
            self.draw_preview()
            return
        x, y = event.x, event.y
        if self.canvas_pts:
            lx, ly = self.canvas_pts[-1]
            if (x - lx) ** 2 + (y - ly) ** 2 < 9:
                return
        self.canvas_pts.append((x, y))
        self.draw_preview()

    def on_release(self, event):
        if not self.drawing:
            return
        self.drawing = False
        if self.mode.get() == "circle":
            mx, my = self.canvas_to_img(event.x, event.y)
            cx, cy, _ = self.circle_info
            r = max(1, int(math.hypot(mx - cx, my - cy)))
            self.roi_circle = (cx, cy, r)
            self.preview_radius = None
            self.canvas_pts = []
            self.refresh()
            self.maybe_auto_export()
            return
        if (event.x, event.y) != self.canvas_pts[-1]:
            self.canvas_pts.append((event.x, event.y))
        if len(self.canvas_pts) < 3:
            self.roi_pts = None
            self.canvas_pts = []
            self.refresh()
            return
        img_pts = [self.canvas_to_img(x, y) for x, y in self.canvas_pts]
        if img_pts[0] != img_pts[-1]:
            img_pts.append(img_pts[0])
        self.roi_pts = np.array(img_pts, dtype=np.int32)
        self.canvas_pts = []
        self.refresh()
        self.maybe_auto_export()

    def clear_roi(self, refresh=True):
        self.roi_pts = None
        self.roi_circle = None
        self.canvas_pts = []
        self.preview_radius = None
        self.drawing = False
        if not refresh:
            return
        if self.vis is not None:
            self.refresh()
        else:
            self.set_hint()

    def current_row(self):
        if self.counts is None or not self.paths:
            return None
        path = self.paths[self.idx]
        c = self.counts
        st = self.roi_stats()
        if self.roi_circle is not None:
            cx, cy, r = self.roi_circle
            roi_type = "circle"
            roi_detail = f"cx={cx};cy={cy};r={r}"
        elif self.roi_pts is not None and len(self.roi_pts) >= 3:
            roi_type = "curve"
            roi_detail = f"points={len(self.roi_pts)}"
        else:
            roi_type = ""
            roi_detail = ""
        return {
            "filename": os.path.basename(path),
            "path": os.path.abspath(path),
            "mode": self.mode.get(),
            "full_red": c["big"],
            "full_green": c["small"],
            "full_blue": c["tiny"],
            "full_total": c["summary"],
            "roi_red": "" if st is None else st["red"],
            "roi_green": "" if st is None else st["green"],
            "roi_blue": "" if st is None else st["blue"],
            "roi_total": "" if st is None else st["total"],
            "roi_type": roi_type,
            "roi_detail": roi_detail,
        }

    def append_csv(self, path=None, silent=False):
        row = self.current_row()
        if row is None:
            if not silent:
                messagebox.showwarning("提示", "请先打开并完成计数")
            return False
        if not self.has_roi():
            if not silent:
                messagebox.showwarning("提示", "请先完成选区")
            return False
        path = path or self.csv_path
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        try:
            with open(path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except OSError as e:
            if not silent:
                messagebox.showerror("导出失败", str(e))
            return False
        if not silent:
            messagebox.showinfo(
                "导出完成",
                f"已写入\n{path}\n"
                f"全图 红={row['full_red']} 绿={row['full_green']} 蓝={row['full_blue']}\n"
                f"选区 红={row['roi_red']} 绿={row['roi_green']} 蓝={row['roi_blue']}",
            )
        else:
            name = os.path.basename(path)
            cur = self.info.cget("text")
            if "已写入" not in cur:
                self.info.config(text=cur + f"  |  已写入 {name}")
        return True

    def maybe_auto_export(self):
        if self.auto_csv.get():
            self.append_csv(silent=True)

    def export_csv(self):
        path = filedialog.asksaveasfilename(
            title="导出统计CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=os.path.basename(self.csv_path),
            initialdir=os.path.dirname(self.csv_path),
        )
        if not path:
            return
        self.csv_path = path
        self.append_csv(path=path, silent=False)

    def run(self):
        self.root.mainloop()


def collect_paths(src):
    if src is None:
        return []
    if os.path.isfile(src):
        return [src]
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"):
        paths.extend(glob.glob(os.path.join(src, "**", ext), recursive=True))
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", nargs="?", default=None)
    args = parser.parse_args()
    app = PlateCounterGUI(collect_paths(args.src))
    app.run()


if __name__ == "__main__":
    main()
