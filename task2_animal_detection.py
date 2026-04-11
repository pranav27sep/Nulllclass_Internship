import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import os
import threading
import time
import numpy as np
from PIL import Image, ImageTk
import torch
import torchvision
from torchvision.models.detection import (fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights)
import torchvision.transforms.functional as TF

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COCO_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
    "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# Animals present in COCO and their diet classification
ANIMAL_COCO_IDS = {
    16: "bird",   17: "cat",   18: "dog",   19: "horse",
    20: "sheep",  21: "cow",   22: "elephant", 23: "bear",
    24: "zebra",  25: "giraffe",
}

CARNIVORES = {"cat", "dog", "bear"}          # Red
HERBIVORES = {"horse", "sheep", "cow", "elephant", "zebra", "giraffe", "bird"}  # Blue

CONF_THRESHOLD = 0.45


# ── Model wrapper ─────────────────────────────────────────────
class AnimalDetector:
    def __init__(self):
        self.model = None
        self.loaded = False

    def load(self, progress_cb=None):
        if progress_cb:
            progress_cb("Loading Faster R-CNN model…")
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        self.model.to(DEVICE).eval()
        self.loaded = True
        if progress_cb:
            progress_cb("Model loaded ✅")

    def predict(self, frame_bgr):
        """Returns list of dicts: {label, is_carnivore, box, score}"""
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = TF.to_tensor(img_rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            preds = self.model(tensor)[0]
        results = []
        for label_id, score, box in zip(
            preds["labels"].cpu().numpy(),
            preds["scores"].cpu().numpy(),
            preds["boxes"].cpu().numpy(),
        ):
            if score < CONF_THRESHOLD:
                continue
            if label_id not in ANIMAL_COCO_IDS:
                continue
            name = ANIMAL_COCO_IDS[label_id]
            is_carn = name in CARNIVORES
            results.append({
                "label": name,
                "is_carnivore": is_carn,
                "box": box.astype(int),
                "score": float(score),
            })
        return results

    def annotate(self, frame_bgr, detections):
        out = frame_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            color = (0, 0, 220) if det["is_carnivore"] else (220, 80, 0)
            tag   = "⚠ CARNIVORE" if det["is_carnivore"] else det["label"]
            label = f"{tag}: {det['label']} {det['score']:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(out, (x1, y1-26), (x2, y1), color, -1)
            cv2.putText(out, label, (x1+4, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return out


# ── GUI ───────────────────────────────────────────────────────
class AnimalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🐾 Animal Detection System — PyTorch")
        self.root.geometry("1100x750")
        self.root.configure(bg="#0D1B2A")

        self.detector = AnimalDetector()
        self.cap = None
        self.running = False
        self.video_thread = None
        self.last_dets = []

        self._build_ui()
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        self.detector.load(progress_cb=lambda m: self.status_bar.config(text=m))

    # ── UI ────────────────────────────────────────────────────
    def _build_ui(self):
        # Top
        top = tk.Frame(self.root, bg="#0A1628", height=55)
        top.pack(fill=tk.X)
        tk.Label(top, text="🐾 ANIMAL DETECTION  &  CLASSIFICATION",
                 font=("Georgia", 16, "bold"), fg="#FFD166", bg="#0A1628").pack(side=tk.LEFT, padx=16, pady=10)

        # Main
        main = tk.Frame(self.root, bg="#0D1B2A")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Canvas / preview
        left = tk.Frame(main, bg="#0A1628")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(left, text="PREVIEW", font=("Georgia", 11, "bold"),
                 fg="#FFD166", bg="#0A1628").pack(pady=(6, 2))
        self.canvas = tk.Label(left, bg="#050F1A", width=700, height=480)
        self.canvas.pack(padx=6, pady=4, fill=tk.BOTH, expand=True)

        # Right panel
        right = tk.Frame(main, bg="#0A1628", width=340)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))
        right.pack_propagate(False)

        # Controls
        ctrl = tk.LabelFrame(right, text=" INPUT SOURCE ", bg="#0A1628",
                              fg="#FFD166", font=("Georgia", 10, "bold"), bd=1)
        ctrl.pack(fill=tk.X, padx=8, pady=8)
        self._btn(ctrl, "📷 Open Image", self.open_image, "#EF476F").pack(fill=tk.X, padx=8, pady=4)
        self._btn(ctrl, "🎬 Open Video", self.open_video, "#118AB2").pack(fill=tk.X, padx=8, pady=4)
        self._btn(ctrl, "📹 Use Webcam", self.use_webcam, "#06D6A0").pack(fill=tk.X, padx=8, pady=4)
        self._btn(ctrl, "⏹ Stop",       self.stop_all,   "#555").pack(fill=tk.X, padx=8, pady=4)

        # Stats
        stats = tk.LabelFrame(right, text=" DETECTION STATS ", bg="#0A1628",
                               fg="#FFD166", font=("Georgia", 10, "bold"), bd=1)
        stats.pack(fill=tk.X, padx=8, pady=8)

        self.lbl_total = self._stat(stats, "Total Animals", "#FFD166")
        self.lbl_carn  = self._stat(stats, "Carnivores 🔴", "#EF476F")
        self.lbl_herb  = self._stat(stats, "Others 🔵",     "#06D6A0")

        # Legend
        leg = tk.LabelFrame(right, text=" LEGEND ", bg="#0A1628",
                             fg="#FFD166", font=("Georgia", 10, "bold"), bd=1)
        leg.pack(fill=tk.X, padx=8, pady=8)
        tk.Label(leg, text="🔴 RED = Carnivore (cat, dog, bear…)",
                 bg="#0A1628", fg="#EF476F", font=("Georgia", 9)).pack(anchor=tk.W, padx=8, pady=2)
        tk.Label(leg, text="🔵 BLUE = Herbivore / Other animals",
                 bg="#0A1628", fg="#06D6A0", font=("Georgia", 9)).pack(anchor=tk.W, padx=8, pady=2)

        # Detection list
        tk.Label(right, text="DETECTED ANIMALS", font=("Georgia", 10, "bold"),
                 fg="#FFD166", bg="#0A1628").pack(pady=(8, 2))
        self.det_list = tk.Listbox(right, bg="#050F1A", fg="white",
                                   font=("Courier", 10), height=12,
                                   selectbackground="#118AB2")
        self.det_list.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Status
        self.status_bar = tk.Label(self.root, text="Loading model…", fg="#A8DADC",
                                   bg="#0A1628", font=("Courier", 10), anchor=tk.W)
        self.status_bar.pack(fill=tk.X, padx=10, pady=4)

    def _btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd, bg=color, fg="white",
                         font=("Georgia", 10, "bold"), relief=tk.FLAT, pady=6,
                         activebackground="#333", cursor="hand2")

    def _stat(self, parent, label, color):
        f = tk.Frame(parent, bg="#0A1628"); f.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(f, text=label, bg="#0A1628", fg="#A8DADC",
                 font=("Courier", 9), width=18, anchor=tk.W).pack(side=tk.LEFT)
        lbl = tk.Label(f, text="0", bg="#0A1628", fg=color,
                       font=("Courier", 14, "bold"), width=4)
        lbl.pack(side=tk.RIGHT)
        return lbl

    # ── Image ─────────────────────────────────────────────────
    def open_image(self):
        if not self.detector.loaded:
            messagebox.showinfo("Wait", "Model still loading…"); return
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path: return
        self.stop_all()
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "Cannot read image."); return
        self._process_and_show(frame)

    def _process_and_show(self, frame):
        dets = self.detector.predict(frame)
        out  = self.detector.annotate(frame, dets)
        self.last_dets = dets
        self._update_stats(dets)
        self._show(out)
        carnivore_count = sum(1 for d in dets if d["is_carnivore"])
        if carnivore_count:
            names = [d["label"] for d in dets if d["is_carnivore"]]
            messagebox.showwarning(
                "⚠ Carnivore Alert!",
                f"{carnivore_count} carnivore(s) detected:\n" + "\n".join(f"• {n.title()}" for n in names)
            )

    # ── Video / Webcam ─────────────────────────────────────────
    def open_video(self):
        if not self.detector.loaded:
            messagebox.showinfo("Wait", "Model still loading…"); return
        path = filedialog.askopenfilename(
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv")])
        if not path: return
        self.stop_all()
        self.cap = cv2.VideoCapture(path)
        self.running = True
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

    def use_webcam(self):
        if not self.detector.loaded:
            messagebox.showinfo("Wait", "Model still loading…"); return
        self.stop_all()
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

    def stop_all(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def _video_loop(self):
        carnivore_alert_shown = False
        frame_skip = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            frame_skip += 1
            if frame_skip % 3 != 0:   # process every 3rd frame
                continue
            dets = self.detector.predict(frame)
            out  = self.detector.annotate(frame, dets)
            self.last_dets = dets
            self.root.after(0, self._update_stats, dets)
            self.root.after(0, self._show, out)
            carn_count = sum(1 for d in dets if d["is_carnivore"])
            if carn_count > 0 and not carnivore_alert_shown:
                carnivore_alert_shown = True
                names = [d["label"] for d in dets if d["is_carnivore"]]
                self.root.after(0, lambda n=names, c=carn_count: messagebox.showwarning(
                    "⚠ Carnivore Alert!",
                    f"{c} carnivore(s) detected:\n" + "\n".join(f"• {x.title()}" for x in n)
                ))
            time.sleep(0.01)

    # ── Helpers ───────────────────────────────────────────────
    def _show(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        max_w, max_h = 700, 480
        scale = min(max_w / w, max_h / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        small = cv2.resize(frame_bgr, (nw, nh))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.configure(image=photo)
        self.canvas.image = photo

    def _update_stats(self, dets):
        total = len(dets)
        carn  = sum(1 for d in dets if d["is_carnivore"])
        herb  = total - carn
        self.lbl_total.config(text=str(total))
        self.lbl_carn.config(text=str(carn))
        self.lbl_herb.config(text=str(herb))
        self.det_list.delete(0, tk.END)
        for d in dets:
            tag = "🔴" if d["is_carnivore"] else "🔵"
            self.det_list.insert(tk.END, f"  {tag}  {d['label'].title()}  ({d['score']:.0%})")
        self.status_bar.config(text=f"Detected {total} animal(s)  |  {carn} carnivore(s)  |  {herb} other(s)")

    def on_close(self):
        self.stop_all()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = AnimalApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
