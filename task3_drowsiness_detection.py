import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import os
import threading
import time
import math
import numpy as np
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EAR_THRESHOLD = 0.25
CONSEC_FRAMES  = 3      

class AgePredictor(nn.Module):
    """MobileNetV2 backbone → age regression (0–100)."""
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        base.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base.last_channel, 1),
            nn.ReLU()
        )
        self.net = base

    def forward(self, x):
        return self.net(x).squeeze(1) * 100.0  


class EyeStateModel(nn.Module):
    """Small CNN: Open(0) / Closed(1)."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class DrowsinessDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade  = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml")
        self.age_model  = AgePredictor().to(DEVICE).eval()
        self.eye_model  = EyeStateModel().to(DEVICE).eval()
        self.frame_counters: dict = {}   

        self.age_tf = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        self.eye_tf = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def _predict_age(self, face_bgr):
        try:
            rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            t   = self.age_tf(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                age = float(self.age_model(t).item())

            return max(5, min(90, int(age)))
        except Exception:
            return 25   

    def _eye_state(self, eye_roi_gray):
        """0 = Open, 1 = Closed (using model)."""
        try:
            pil = Image.fromarray(eye_roi_gray)
            t   = self.eye_tf(pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = self.eye_model(t)
                return int(out.argmax(1).item())
        except Exception:
            return 0

    def _ear_from_eyes(self, eyes, face_w, face_h):
        """Fallback: estimate EAR from eye bounding box aspect ratio."""
        if len(eyes) == 0:
            return 0.15  
        ears = []
        for (ex, ey, ew, eh) in eyes:
            ear = eh / max(ew, 1)
            ears.append(ear)
        return float(np.mean(ears))

    def analyze(self, frame_bgr):
        """Returns list of dicts: {box, age, sleeping, eye_state}."""
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        results = []
        for idx, (x, y, w, h) in enumerate(faces):
            face_bgr  = frame_bgr[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]
            age = self._predict_age(face_bgr)

            eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 3, minSize=(20, 20))
            ear  = self._ear_from_eyes(eyes, w, h)
            is_sleeping = ear < EAR_THRESHOLD

            fid = idx
            if is_sleeping:
                self.frame_counters[fid] = self.frame_counters.get(fid, 0) + 1
            else:
                self.frame_counters[fid] = 0
            sleeping_confirmed = self.frame_counters.get(fid, 0) >= CONSEC_FRAMES

            results.append({
                "box": (x, y, w, h),
                "age": age,
                "sleeping": sleeping_confirmed,
                "ear": ear,
            })
        return results

    def annotate(self, frame_bgr, detections):
        out = frame_bgr.copy()
        for det in detections:
            x, y, w, h = det["box"]
            if det["sleeping"]:
                color = (0, 0, 220)    
                status = "SLEEPING 😴"
            else:
                color = (0, 200, 80)   
                status = "Awake ✓"
            label = f"{status} | Age: ~{det['age']}"
            cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(out, (x, y-30), (x+w, y), color, -1)
            cv2.putText(out, label, (x+4, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return out

class DrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("😴 Drowsiness Detection System — PyTorch")
        self.root.geometry("1100x750")
        self.root.configure(bg="#1C1C2E")

        self.detector = DrowsinessDetector()
        self.cap = None
        self.running = False
        self.popup_shown = False
        self._build_ui()

    def _build_ui(self):
        top = tk.Frame(self.root, bg="#12122A", height=55)
        top.pack(fill=tk.X)
        tk.Label(top, text="😴 DRIVER / PASSENGER DROWSINESS DETECTION",
                 font=("Helvetica", 15, "bold"), fg="#F72585", bg="#12122A").pack(
                 side=tk.LEFT, padx=16, pady=10)

        main = tk.Frame(self.root, bg="#1C1C2E")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        left = tk.Frame(main, bg="#12122A")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(left, text="LIVE PREVIEW", font=("Helvetica", 11, "bold"),
                 fg="#F72585", bg="#12122A").pack(pady=(6,2))
        self.preview = tk.Label(left, bg="#07071A", width=680, height=480)
        self.preview.pack(padx=6, pady=4, fill=tk.BOTH, expand=True)

        right = tk.Frame(main, bg="#12122A", width=340)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(8,0))
        right.pack_propagate(False)

        ctl = tk.LabelFrame(right, text=" INPUT ", bg="#12122A",
                            fg="#F72585", font=("Helvetica", 10, "bold"))
        ctl.pack(fill=tk.X, padx=8, pady=8)
        self._btn(ctl, "📷 Open Image", self.open_image, "#7B2D8B").pack(fill=tk.X, padx=8, pady=4)
        self._btn(ctl, "🎬 Open Video", self.open_video, "#0077B6").pack(fill=tk.X, padx=8, pady=4)
        self._btn(ctl, "📹 Webcam",     self.use_webcam, "#2D6A4F").pack(fill=tk.X, padx=8, pady=4)
        self._btn(ctl, "⏹ Stop",        self.stop_all,   "#555").pack(fill=tk.X, padx=8, pady=4)

        st = tk.LabelFrame(right, text=" STATISTICS ", bg="#12122A",
                           fg="#F72585", font=("Helvetica", 10, "bold"))
        st.pack(fill=tk.X, padx=8, pady=8)
        self.lbl_people   = self._stat(st, "Total People",   "#A8DADC")
        self.lbl_sleeping = self._stat(st, "Sleeping 🔴",    "#F72585")
        self.lbl_awake    = self._stat(st, "Awake  🟢",      "#06D6A0")

        tk.Label(right, text="DETECTED PERSONS", font=("Helvetica", 10, "bold"),
                 fg="#F72585", bg="#12122A").pack(pady=(8,2))
        self.listbox = tk.Listbox(right, bg="#07071A", fg="white",
                                  font=("Courier", 10), height=14,
                                  selectbackground="#7B2D8B")
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.status = tk.Label(self.root, text="Ready", bg="#12122A",
                               fg="#A8DADC", font=("Courier", 10), anchor=tk.W)
        self.status.pack(fill=tk.X, padx=10, pady=4)

    def _btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd, bg=color, fg="white",
                         font=("Helvetica", 9, "bold"), relief=tk.FLAT, pady=6,
                         activebackground="#333", cursor="hand2")

    def _stat(self, parent, label, color):
        f = tk.Frame(parent, bg="#12122A"); f.pack(fill=tk.X, padx=8, pady=3)
        tk.Label(f, text=label, bg="#12122A", fg="#A8DADC",
                 font=("Courier", 9), width=16, anchor=tk.W).pack(side=tk.LEFT)
        lbl = tk.Label(f, text="0", bg="#12122A", fg=color,
                       font=("Courier", 16, "bold"), width=4)
        lbl.pack(side=tk.RIGHT)
        return lbl

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        self.stop_all()
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "Cannot read image."); return
        dets = self.detector.analyze(frame)
        out  = self.detector.annotate(frame, dets)
        self._show(out)
        self._update_ui(dets, popup=True)

    def open_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv")])
        if not path: return
        self.stop_all()
        self.cap = cv2.VideoCapture(path)
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def use_webcam(self):
        self.stop_all()
        self.cap = cv2.VideoCapture(0)
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop_all(self):
        self.running = False
        self.popup_shown = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def _loop(self):
        skip = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            skip += 1
            if skip % 4 != 0:
                continue
            dets = self.detector.analyze(frame)
            out  = self.detector.annotate(frame, dets)
            self.root.after(0, self._show, out)
            self.root.after(0, self._update_ui, dets, not self.popup_shown)
            if any(d["sleeping"] for d in dets):
                self.popup_shown = True
            time.sleep(0.01)

    def _show(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = min(680/w, 480/h, 1.0)
        small = cv2.resize(frame_bgr, (int(w*scale), int(h*scale)))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview.configure(image=photo)
        self.preview.image = photo

    def _update_ui(self, dets, popup=False):
        total    = len(dets)
        sleeping = [d for d in dets if d["sleeping"]]
        awake    = total - len(sleeping)
        self.lbl_people.config(text=str(total))
        self.lbl_sleeping.config(text=str(len(sleeping)))
        self.lbl_awake.config(text=str(awake))
        self.listbox.delete(0, tk.END)
        for i, det in enumerate(dets, 1):
            icon = "🔴 SLEEPING" if det["sleeping"] else "🟢 Awake"
            self.listbox.insert(tk.END, f"  Person {i}: {icon}  |  Age ≈ {det['age']}")
        self.status.config(
            text=f"People: {total}  |  Sleeping: {len(sleeping)}  |  Awake: {awake}")
        if popup and sleeping:
            info = "\n".join(
                f"  Person {i+1}: Age ≈ {d['age']}" for i, d in enumerate(sleeping))
            messagebox.showwarning(
                "⚠ Drowsiness Alert!",
                f"{len(sleeping)} person(s) detected SLEEPING:\n\n{info}\n\n"
                "Please ensure driver safety!")

    def on_close(self):
        self.stop_all()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = DrowsinessApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
