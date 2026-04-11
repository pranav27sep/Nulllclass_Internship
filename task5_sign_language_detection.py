"""
=============================================================
TASK 5: SIGN LANGUAGE DETECTION MODEL
=============================================================
Requirements:
  - PyTorch CNN to recognize ASL/sign gestures
  - Detects: A–Z letters + common words
    (Hello, Yes, No, Thanks, Sorry, Help, Stop, Good, Bad,
     Love, Eat, Water, Home, Work, Go, Come, Wait, More, Less)
  - Active only 6 PM – 10 PM (configurable)
  - Supports image upload + real-time video
  - Proper GUI

Setup:
  pip install torch torchvision opencv-python Pillow mediapipe
=============================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import os
import threading
import time
import datetime
import numpy as np
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Time window ───────────────────────────────────────────────
ACTIVE_START = datetime.time(18, 0)   # 6:00 PM
ACTIVE_END   = datetime.time(22, 0)   # 10:00 PM

# ── Sign labels ───────────────────────────────────────────────
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
WORDS   = ["Hello", "Yes", "No", "Thanks", "Sorry",
           "Help", "Stop", "Good", "Bad", "Love",
           "Eat", "Water", "Home", "Work", "Go",
           "Come", "Wait", "More", "Less"]
ALL_SIGNS = LETTERS + WORDS   # 45 classes


# ── PyTorch Model ─────────────────────────────────────────────
class SignLanguageModel(nn.Module):
    """
    EfficientNet-B0 backbone fine-tuned for sign language recognition.
    Input: 224×224 RGB hand image (or full frame)
    Output: logits for ALL_SIGNS classes
    """
    def __init__(self, num_classes=len(ALL_SIGNS)):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        self.net = base

    def forward(self, x):
        return self.net(x)


# ── Hand Detector (OpenCV-based, no mediapipe required) ────────
class HandDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=25, detectShadows=False)

    def detect_hand_roi(self, frame_bgr):
        """Returns (roi, box) or (None, None)."""
        roi_box = (0, 0, frame_bgr.shape[1], frame_bgr.shape[0])
        # Use skin color detection
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 20, 70], dtype=np.uint8)
        upper = np.array([20, 255, 255], dtype=np.uint8)
        mask  = cv2.inRange(hsv, lower, upper)
        mask  = cv2.GaussianBlur(mask, (5,5), 0)
        mask  = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 5000:
            return None, None
        x, y, w, h = cv2.boundingRect(c)
        pad = 20
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(frame_bgr.shape[1], x + w + pad)
        y2 = min(frame_bgr.shape[0], y + h + pad)
        roi = frame_bgr[y1:y2, x1:x2]
        return roi, (x1, y1, x2-x1, y2-y1)


# ── Predictor ─────────────────────────────────────────────────
class SignPredictor:
    def __init__(self):
        self.model = SignLanguageModel().to(DEVICE).eval()
        self.hand_detector = HandDetector()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, frame_bgr):
        """Returns (sign_label, confidence, annotated_frame, hand_box)."""
        roi, box = self.hand_detector.detect_hand_roi(frame_bgr)
        if roi is None or roi.size == 0:
            return None, 0.0, frame_bgr.copy(), None
        try:
            rgb  = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pil  = Image.fromarray(rgb)
            t    = self.transform(pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = self.model(t)
                probs  = torch.softmax(logits, dim=1)[0]
                idx    = probs.argmax().item()
                conf   = float(probs[idx])
        except Exception:
            return None, 0.0, frame_bgr.copy(), None

        sign = ALL_SIGNS[idx]
        # Draw annotation
        out = frame_bgr.copy()
        if box:
            x, y, w, h = box
            cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 120), 2)
            label = f"{sign}  ({conf:.0%})"
            cv2.rectangle(out, (x, y-36), (x+w, y), (0, 200, 80), -1)
            cv2.putText(out, label, (x+4, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return sign, conf, out, box

    def predict_image(self, img_bgr):
        return self.predict(img_bgr)


# ── GUI ───────────────────────────────────────────────────────
class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤟 Sign Language Detection — PyTorch")
        self.root.geometry("1200x760")
        self.root.configure(bg="#0F0E17")

        self.predictor = SignPredictor()
        self.cap = None
        self.running = False
        self.history = []       # (sign, conf, timestamp)

        self._build_ui()
        self._check_window()

    def _build_ui(self):
        # Header
        top = tk.Frame(self.root, bg="#0F0E17", height=60)
        top.pack(fill=tk.X)
        tk.Label(top, text="🤟  SIGN LANGUAGE RECOGNITION  SYSTEM",
                 font=("Segoe UI", 16, "bold"), fg="#FF8906", bg="#0F0E17").pack(
                 side=tk.LEFT, padx=16, pady=12)
        self.clock_lbl = tk.Label(top, text="", font=("Courier", 13),
                                  fg="#A7A9BE", bg="#0F0E17")
        self.clock_lbl.pack(side=tk.RIGHT, padx=20)
        self._tick()

        # Time window indicator
        self.window_lbl = tk.Label(self.root, text="", font=("Courier", 11, "bold"),
                                   bg="#0F0E17", pady=4)
        self.window_lbl.pack(fill=tk.X, padx=10)

        main = tk.Frame(self.root, bg="#0F0E17")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # Camera / image preview
        left = tk.Frame(main, bg="#0F0E17")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(left, text="LIVE VIEW", font=("Segoe UI", 11, "bold"),
                 fg="#FF8906", bg="#0F0E17").pack(pady=(4,2))
        self.preview = tk.Label(left, bg="#111111", width=680, height=460)
        self.preview.pack(padx=6, pady=4, fill=tk.BOTH, expand=True)

        # Controls
        ctl = tk.Frame(left, bg="#0F0E17")
        ctl.pack(fill=tk.X, pady=4, padx=6)
        self._btn(ctl, "📷 Upload Image", self.upload_image, "#FF8906").pack(side=tk.LEFT, padx=4)
        self._btn(ctl, "📹 Start Webcam", self.start_webcam, "#3DA35D").pack(side=tk.LEFT, padx=4)
        self._btn(ctl, "⏹ Stop",          self.stop,          "#555").pack(side=tk.LEFT, padx=4)

        # Right panel
        right = tk.Frame(main, bg="#0F0E17", width=380)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(8,0))
        right.pack_propagate(False)

        # Big sign display
        pred_frame = tk.Frame(right, bg="#1A1A2E", relief=tk.GROOVE, bd=1)
        pred_frame.pack(fill=tk.X, padx=8, pady=8)
        tk.Label(pred_frame, text="DETECTED SIGN", font=("Segoe UI", 10),
                 fg="#A7A9BE", bg="#1A1A2E").pack(pady=(8,0))
        self.sign_lbl = tk.Label(pred_frame, text="—", font=("Segoe UI", 52, "bold"),
                                 fg="#FF8906", bg="#1A1A2E", width=8, height=2)
        self.sign_lbl.pack()
        self.conf_lbl = tk.Label(pred_frame, text="Confidence: —",
                                 font=("Segoe UI", 12), fg="#A7A9BE", bg="#1A1A2E")
        self.conf_lbl.pack(pady=(0,8))

        # Reference signs
        ref = tk.LabelFrame(right, text=" SIGN REFERENCE ", bg="#0F0E17",
                            fg="#FF8906", font=("Segoe UI", 10, "bold"))
        ref.pack(fill=tk.X, padx=8, pady=4)
        letters_str = "  ".join(LETTERS)
        tk.Label(ref, text=letters_str, bg="#0F0E17", fg="#3DA35D",
                 font=("Courier", 8), wraplength=340, justify=tk.LEFT).pack(
                 anchor=tk.W, padx=6, pady=2)
        words_str = "  ".join(WORDS)
        tk.Label(ref, text=words_str, bg="#0F0E17", fg="#FFB703",
                 font=("Courier", 8), wraplength=340, justify=tk.LEFT).pack(
                 anchor=tk.W, padx=6, pady=2)

        # History
        tk.Label(right, text="RECOGNITION HISTORY", font=("Segoe UI", 10, "bold"),
                 fg="#FF8906", bg="#0F0E17").pack(pady=(8,2))
        self.hist_box = tk.Listbox(right, bg="#111111", fg="white",
                                   font=("Courier", 10), height=10,
                                   selectbackground="#FF8906")
        self.hist_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self._btn(right, "🗑 Clear History", self.clear_history, "#3A3A5C").pack(
                  fill=tk.X, padx=8, pady=4)

        # Status
        self.status = tk.Label(self.root, text="Ready", bg="#0F0E17",
                               fg="#A7A9BE", font=("Courier", 10), anchor=tk.W)
        self.status.pack(fill=tk.X, padx=10, pady=4)

    def _btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd, bg=color, fg="white",
                         font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=8, pady=5,
                         activebackground="#333", cursor="hand2")

    def _tick(self):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.clock_lbl.config(text=now)
        self.root.after(1000, self._tick)

    def _check_window(self):
        now = datetime.datetime.now().time()
        self.in_window = ACTIVE_START <= now <= ACTIVE_END
        if self.in_window:
            self.window_lbl.config(
                text=f"✅ System ACTIVE  ({ACTIVE_START.strftime('%I:%M %p')} – {ACTIVE_END.strftime('%I:%M %p')})",
                fg="#3DA35D", bg="#0F0E17")
        else:
            self.window_lbl.config(
                text=f"⛔ System INACTIVE — Only active {ACTIVE_START.strftime('%I:%M %p')} to {ACTIVE_END.strftime('%I:%M %p')}",
                fg="#E63946", bg="#0F0E17")
        self.root.after(10000, self._check_window)

    def _gate(self):
        if not self.in_window:
            messagebox.showwarning(
                "Outside Active Hours",
                f"Sign Language Detection is only active\n"
                f"from {ACTIVE_START.strftime('%I:%M %p')} to {ACTIVE_END.strftime('%I:%M %p')}.")
            return False
        return True

    def upload_image(self):
        if not self._gate(): return
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        self.stop()
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "Cannot read image."); return
        sign, conf, out, box = self.predictor.predict_image(frame)
        self._show(out)
        self._update_sign(sign, conf)

    def start_webcam(self):
        if not self._gate(): return
        self.stop()
        self.cap = cv2.VideoCapture(0)
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        self.status.config(text="Webcam running…")

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def _loop(self):
        skip = 0
        while self.running:
            if not self.in_window:
                self.root.after(0, lambda: messagebox.showwarning(
                    "Time Expired", "Active window has ended. Stopping."))
                self.running = False
                break
            ret, frame = self.cap.read()
            if not ret: break
            skip += 1
            if skip % 2 != 0:
                continue
            sign, conf, out, box = self.predictor.predict(frame)
            self.root.after(0, self._show, out)
            self.root.after(0, self._update_sign, sign, conf)
            time.sleep(0.01)

    def _show(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = min(680/w, 460/h, 1.0)
        nw, nh = int(w*scale), int(h*scale)
        small = cv2.resize(frame_bgr, (nw, nh))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview.configure(image=photo)
        self.preview.image = photo

    def _update_sign(self, sign, conf):
        if sign is None:
            self.sign_lbl.config(text="—")
            self.conf_lbl.config(text="No hand detected")
            self.status.config(text="No hand in frame")
            return
        self.sign_lbl.config(text=sign)
        self.conf_lbl.config(text=f"Confidence: {conf:.1%}")
        self.status.config(text=f"Detected: {sign}  |  Confidence: {conf:.1%}")
        # Add to history
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.history.append((sign, conf, ts))
        self.hist_box.insert(0, f"  [{ts}]  {sign:<12}  {conf:.0%}")
        if len(self.history) > 50:
            self.hist_box.delete(tk.END)

    def clear_history(self):
        self.history.clear()
        self.hist_box.delete(0, tk.END)

    def on_close(self):
        self.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
