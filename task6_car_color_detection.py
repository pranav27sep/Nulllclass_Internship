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
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
import torchvision.transforms.functional as TF

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAR_CLASSES    = {3: "car", 6: "bus", 8: "truck", 4: "motorcycle", 2: "bicycle"}
PERSON_CLASS   = 1
CONF_THRESHOLD = 0.45

COLOR_RANGES = {
    "Blue":   ([100, 50, 50],  [135, 255, 255]),
    "Red":    ([0, 80, 80],    [10, 255, 255]),
    "Red2":   ([170, 80, 80],  [180, 255, 255]),
    "White":  ([0, 0, 180],    [180, 30, 255]),
    "Black":  ([0, 0, 0],      [180, 255, 50]),
    "Silver": ([0, 0, 140],    [180, 30, 210]),
    "Yellow": ([20, 100, 100], [35, 255, 255]),
    "Green":  ([35, 50, 50],   [85, 255, 255]),
    "Gray":   ([0, 0, 60],     [180, 30, 140]),
    "Orange": ([10, 100, 100], [20, 255, 255]),
}

def detect_car_color(frame_bgr, box):
    """Determine dominant color of vehicle from its bounding box region."""
    x1, y1, x2, y2 = box
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return "Unknown"
      
    h = roi.shape[0]
    roi = roi[:int(h * 0.6)]
    if roi.size == 0:
        return "Unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    scores = {}
    for name, (lo, hi) in COLOR_RANGES.items():
        mask  = cv2.inRange(hsv, np.array(lo, dtype=np.uint8),
                            np.array(hi, dtype=np.uint8))
        score = int(mask.sum() / 255)
        scores[name] = score
    
    scores["Red"] = scores.get("Red", 0) + scores.pop("Red2", 0)
    best = max(scores, key=scores.get)
    return best if scores[best] > 50 else "Unknown"

class TrafficDetector:
    def __init__(self):
        self.model = None
        self.loaded = False

    def load(self, cb=None):
        if cb: cb("Loading Faster R-CNN model…")
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        self.model.to(DEVICE).eval()
        self.loaded = True
        if cb: cb("Model loaded ✅")

    def detect(self, frame_bgr):
        """Returns cars list and people list."""
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = TF.to_tensor(rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            preds = self.model(tensor)[0]
        cars, people = [], []
        for label, score, box in zip(
            preds["labels"].cpu().numpy(),
            preds["scores"].cpu().numpy(),
            preds["boxes"].cpu().numpy(),
        ):
            if score < CONF_THRESHOLD:
                continue
            box = box.astype(int)
            if label in CAR_CLASSES:
                color = detect_car_color(frame_bgr, box)
                cars.append({
                    "type": CAR_CLASSES[label],
                    "color": color,
                    "box": box,
                    "score": float(score),
                    "is_blue": color == "Blue",
                })
            elif label == PERSON_CLASS:
                people.append({"box": box, "score": float(score)})
        return cars, people

    def annotate(self, frame_bgr, cars, people):
        out = frame_bgr.copy()
        for car in cars:
            x1, y1, x2, y2 = car["box"]
            box_color  = (0, 0, 220) if car["is_blue"] else (220, 80, 0)
            label = f"{car['color']} {car['type']} ({car['score']:.0%})"
            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
            cv2.rectangle(out, (x1, y1-28), (x2, y1), box_color, -1)
            cv2.putText(out, label, (x1+4, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        for p in people:
            x1, y1, x2, y2 = p["box"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 180), 2)
            cv2.putText(out, "Person", (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 180), 1)
        total_cars = len(cars)
        blue_cars  = sum(1 for c in cars if c["is_blue"])
        total_ppl  = len(people)
        hud_lines  = [
            f"Total Vehicles : {total_cars}",
            f"Blue Cars      : {blue_cars}",
            f"People at Signal: {total_ppl}",
        ]
        overlay = out.copy()
        cv2.rectangle(overlay, (8, 8), (280, 90), (0, 0, 0), -1)
        out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
        for i, line in enumerate(hud_lines):
            color = (0, 0, 220) if "Blue" in line else (220, 220, 220)
            cv2.putText(out, line, (14, 30 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        return out

class CarColorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 Car Colour Detection — PyTorch")
        self.root.geometry("1200x760")
        self.root.configure(bg="#0B0C10")

        self.detector = TrafficDetector()
        self.cap = None
        self.running = False
        self.color_counts: dict = {}

        self._build_ui()
        threading.Thread(target=lambda: self.detector.load(
            cb=lambda m: self.root.after(0, self.status.config, {"text": m})
        ), daemon=True).start()

    def _build_ui(self):
        top = tk.Frame(self.root, bg="#1F2833", height=58)
        top.pack(fill=tk.X)
        tk.Label(top, text="🚗  CAR COLOUR DETECTION  &  TRAFFIC MONITOR",
                 font=("Impact", 16), fg="#66FCF1", bg="#1F2833").pack(
                 side=tk.LEFT, padx=16, pady=12)

        main = tk.Frame(self.root, bg="#0B0C10")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        left = tk.Frame(main, bg="#0B0C10")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(left, text="TRAFFIC PREVIEW", font=("Impact", 12),
                 fg="#66FCF1", bg="#0B0C10").pack(pady=(4,2))
        self.preview = tk.Label(left, bg="#0D1117", width=700, height=480)
        self.preview.pack(padx=6, pady=4, fill=tk.BOTH, expand=True)

        ctl = tk.Frame(left, bg="#0B0C10")
        ctl.pack(fill=tk.X, padx=8, pady=4)
        self._btn(ctl, "🖼 Open Image", self.open_image, "#C3073F").pack(side=tk.LEFT, padx=4)
        self._btn(ctl, "🎬 Open Video", self.open_video, "#950740").pack(side=tk.LEFT, padx=4)
        self._btn(ctl, "📷 Webcam",    self.use_webcam, "#1A1A2E").pack(side=tk.LEFT, padx=4)
        self._btn(ctl, "⏹ Stop",       self.stop_all,   "#444").pack(side=tk.LEFT, padx=4)

        right = tk.Frame(main, bg="#0B0C10", width=380)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(8,0))
        right.pack_propagate(False)

        st = tk.LabelFrame(right, text=" TRAFFIC STATISTICS ", bg="#0B0C10",
                           fg="#66FCF1", font=("Impact", 11))
        st.pack(fill=tk.X, padx=8, pady=8)
        self.lbl_total  = self._stat(st, "Total Vehicles", "#66FCF1")
        self.lbl_blue   = self._stat(st, "Blue Cars 🔴",    "#FF4757")
        self.lbl_others = self._stat(st, "Other Cars 🔵",   "#4A90E2")
        self.lbl_people = self._stat(st, "People 🟢",       "#2ED573")

        tk.Label(right, text="COLOR BREAKDOWN", font=("Impact", 11),
                 fg="#66FCF1", bg="#0B0C10").pack(pady=(8,2))
        self.color_frame = tk.Frame(right, bg="#0B0C10")
        self.color_frame.pack(fill=tk.X, padx=8)
        self.color_labels: dict = {}

        leg = tk.LabelFrame(right, text=" LEGEND ", bg="#0B0C10",
                            fg="#66FCF1", font=("Impact", 10))
        leg.pack(fill=tk.X, padx=8, pady=8)
        tk.Label(leg, text="🔴 RED box = Blue car (flagged)",
                 bg="#0B0C10", fg="#FF4757", font=("Courier", 9)).pack(anchor=tk.W, padx=8, pady=2)
        tk.Label(leg, text="🔵 BLUE box = Other colored car",
                 bg="#0B0C10", fg="#4A90E2", font=("Courier", 9)).pack(anchor=tk.W, padx=8, pady=2)
        tk.Label(leg, text="🟢 TEAL box = Person at signal",
                 bg="#0B0C10", fg="#2ED573", font=("Courier", 9)).pack(anchor=tk.W, padx=8, pady=2)

        tk.Label(right, text="DETECTED VEHICLES", font=("Impact", 11),
                 fg="#66FCF1", bg="#0B0C10").pack(pady=(8,2))
        self.veh_list = tk.Listbox(right, bg="#0D1117", fg="white",
                                   font=("Courier", 10), height=10,
                                   selectbackground="#C3073F")
        self.veh_list.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.status = tk.Label(self.root, text="Loading model…", bg="#1F2833",
                               fg="#66FCF1", font=("Courier", 10), anchor=tk.W)
        self.status.pack(fill=tk.X, padx=10, pady=4)

    def _btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd, bg=color, fg="white",
                         font=("Impact", 10), relief=tk.FLAT, padx=10, pady=6,
                         activebackground="#555", cursor="hand2")

    def _stat(self, parent, label, color):
        f = tk.Frame(parent, bg="#0B0C10"); f.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(f, text=label, bg="#0B0C10", fg="#A8DADC",
                 font=("Courier", 9), width=18, anchor=tk.W).pack(side=tk.LEFT)
        lbl = tk.Label(f, text="0", bg="#0B0C10", fg=color,
                       font=("Courier", 18, "bold"), width=4)
        lbl.pack(side=tk.RIGHT)
        return lbl

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
        self._process(frame)

    def open_video(self):
        if not self.detector.loaded:
            messagebox.showinfo("Wait", "Model still loading…"); return
        path = filedialog.askopenfilename(
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv")])
        if not path: return
        self.stop_all()
        self.cap = cv2.VideoCapture(path)
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def use_webcam(self):
        if not self.detector.loaded:
            messagebox.showinfo("Wait", "Model still loading…"); return
        self.stop_all()
        self.cap = cv2.VideoCapture(0)
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop_all(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def _process(self, frame):
        cars, people = self.detector.detect(frame)
        out = self.detector.annotate(frame, cars, people)
        self._show(out)
        self._update_ui(cars, people)

    def _loop(self):
        skip = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret: break
            skip += 1
            if skip % 4 != 0: continue
            cars, people = self.detector.detect(frame)
            out = self.detector.annotate(frame, cars, people)
            self.root.after(0, self._show, out)
            self.root.after(0, self._update_ui, cars, people)
            time.sleep(0.01)

    def _show(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = min(700/w, 480/h, 1.0)
        nw, nh = int(w*scale), int(h*scale)
        small = cv2.resize(frame_bgr, (nw, nh))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview.configure(image=photo)
        self.preview.image = photo

    def _update_ui(self, cars, people):
        total  = len(cars)
        blue   = sum(1 for c in cars if c["is_blue"])
        others = total - blue
        ppl    = len(people)
        self.lbl_total.config(text=str(total))
        self.lbl_blue.config(text=str(blue))
        self.lbl_others.config(text=str(others))
        self.lbl_people.config(text=str(ppl))

        counts: dict = {}
        for c in cars:
            counts[c["color"]] = counts.get(c["color"], 0) + 1
        for w in self.color_frame.winfo_children():
            w.destroy()
        for color_name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            icon = "🔴" if color_name == "Blue" else "⬜"
            row = tk.Frame(self.color_frame, bg="#0B0C10"); row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f"  {icon}  {color_name}", bg="#0B0C10",
                     fg="#A8DADC", font=("Courier", 9), width=16,
                     anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(row, text=str(cnt), bg="#0B0C10",
                     fg="#66FCF1", font=("Courier", 10, "bold")).pack(side=tk.RIGHT)

        self.veh_list.delete(0, tk.END)
        for i, c in enumerate(cars, 1):
            flag = "⚠" if c["is_blue"] else " "
            self.veh_list.insert(tk.END,
                f"  {flag} {i}. {c['color']} {c['type']}  ({c['score']:.0%})")

        self.status.config(
            text=f"Vehicles: {total}  |  Blue (flagged): {blue}  |  Others: {others}  |  People: {ppl}")


def main():
    root = tk.Tk()
    app = CarColorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close if hasattr(app, "on_close") else root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
