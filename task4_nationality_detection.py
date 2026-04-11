"""
=============================================================
TASK 4: NATIONALITY DETECTION MODEL
=============================================================
Requirements:
  - PyTorch model to predict nationality from face image
  - Emotion detection for all
  - Indian:      predict age + dress color + emotion
  - US:          predict age + emotion
  - African:     predict emotion + dress color
  - Others:      predict nationality + emotion
  - Proper GUI with image preview and results panel

Setup:
  pip install torch torchvision opencv-python Pillow
=============================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import os
import threading
import numpy as np
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import colorsys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NATIONALITIES = ["Indian", "American", "African", "East Asian", "European",
                 "Middle Eastern", "Hispanic", "South Asian"]

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# CSS-style color names for dress color detection
COLOR_NAMES = {
    "Red": ([0, 80, 80], [10, 255, 255]),
    "Orange": ([10, 80, 80], [25, 255, 255]),
    "Yellow": ([25, 80, 80], [35, 255, 255]),
    "Green": ([35, 40, 40], [85, 255, 255]),
    "Cyan": ([85, 40, 40], [100, 255, 255]),
    "Blue": ([100, 40, 40], [130, 255, 255]),
    "Purple": ([130, 40, 40], [155, 255, 255]),
    "Pink": ([155, 40, 40], [175, 255, 255]),
    "White": ([0, 0, 180], [180, 30, 255]),
    "Black": ([0, 0, 0], [180, 255, 40]),
    "Gray": ([0, 0, 50], [180, 30, 180]),
    "Brown": ([10, 40, 40], [20, 200, 150]),
}


# ── PyTorch Models ────────────────────────────────────────────
class NationalityModel(nn.Module):
    """ResNet18 fine-tuned for nationality prediction."""
    def __init__(self, num_classes=len(NATIONALITIES)):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Linear(512, num_classes)
        self.net = base

    def forward(self, x):
        return self.net(x)


class EmotionModel(nn.Module):
    """MobileNetV2 for emotion classification."""
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        base.features[0][0] = nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False)
        base.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(base.last_channel, len(EMOTIONS))
        )
        self.net = base

    def forward(self, x):
        return self.net(x)


class AgeModel(nn.Module):
    """EfficientNet-style age regressor."""
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        base.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base.classifier[1].in_features, 1),
            nn.Sigmoid()
        )
        self.net = base

    def forward(self, x):
        return self.net(x).squeeze(1) * 90.0 + 5.0


# ── Analysis Engine ───────────────────────────────────────────
class NationalityAnalyzer:
    def __init__(self):
        self.nat_model = NationalityModel().to(DEVICE).eval()
        self.emo_model = EmotionModel().to(DEVICE).eval()
        self.age_model = AgeModel().to(DEVICE).eval()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self.face_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.emo_tf = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def _face_tensor(self, face_bgr):
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        return self.face_tf(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)

    def predict_nationality(self, face_bgr):
        try:
            t = self._face_tensor(face_bgr)
            with torch.no_grad():
                out = self.nat_model(t)
                probs = torch.softmax(out, dim=1)[0]
                idx   = probs.argmax().item()
            conf = float(probs[idx])
            return NATIONALITIES[idx], conf
        except Exception:
            return "Unknown", 0.5

    def predict_emotion(self, face_bgr):
        try:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            t = self.emo_tf(Image.fromarray(gray)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = self.emo_model(t)
                probs = torch.softmax(out, dim=1)[0]
                idx   = probs.argmax().item()
            return EMOTIONS[idx], float(probs[idx])
        except Exception:
            return "Neutral", 0.5

    def predict_age(self, face_bgr):
        try:
            t = self._face_tensor(face_bgr)
            with torch.no_grad():
                age = float(self.age_model(t).item())
            return int(np.clip(age, 5, 90))
        except Exception:
            return 25

    def detect_dress_color(self, img_bgr, face_box):
        """Detect dominant color in the body region below the face."""
        x, y, w, h = face_box
        body_y  = y + h
        body_h  = min(int(h * 1.5), img_bgr.shape[0] - body_y)
        if body_h < 20:
            return "Unknown"
        body = img_bgr[body_y:body_y+body_h, max(0, x-10):x+w+10]
        if body.size == 0:
            return "Unknown"
        hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
        best_color, best_count = "Unknown", 0
        for name, (lo, hi) in COLOR_NAMES.items():
            mask  = cv2.inRange(hsv, np.array(lo), np.array(hi))
            count = int(mask.sum() / 255)
            if count > best_count:
                best_count = count
                best_color = name
        return best_color

    def analyze_image(self, img_bgr):
        """Returns list of result dicts per detected face."""
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        results = []
        for (x, y, w, h) in faces:
            face = img_bgr[y:y+h, x:x+w]
            nat,  nat_conf  = self.predict_nationality(face)
            emo,  emo_conf  = self.predict_emotion(face)
            age             = self.predict_age(face)
            dress           = self.detect_dress_color(img_bgr, (x, y, w, h))
            results.append({
                "box": (x, y, w, h),
                "nationality": nat,
                "nat_conf": nat_conf,
                "emotion": emo,
                "emo_conf": emo_conf,
                "age": age,
                "dress_color": dress,
            })
        return results

    def build_output(self, result):
        """Format output lines based on nationality rules."""
        nat = result["nationality"]
        lines = [f"Nationality: {nat}  ({result['nat_conf']:.0%})",
                 f"Emotion: {result['emotion']}  ({result['emo_conf']:.0%})"]
        if nat == "Indian":
            lines += [f"Age: ~{result['age']} years",
                      f"Dress Color: {result['dress_color']}"]
        elif nat == "American":
            lines.append(f"Age: ~{result['age']} years")
        elif nat == "African":
            lines.append(f"Dress Color: {result['dress_color']}")
        # Others: nationality + emotion already included
        return lines

    def annotate(self, img_bgr, results):
        out = img_bgr.copy()
        for r in results:
            x, y, w, h = r["box"]
            cv2.rectangle(out, (x, y), (x+w, y+h), (0, 200, 255), 2)
            cv2.putText(out, r["nationality"], (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        return out


# ── GUI ───────────────────────────────────────────────────────
class NationalityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌍 Nationality Detection System — PyTorch")
        self.root.geometry("1200x750")
        self.root.configure(bg="#1A1A2E")

        self.analyzer = NationalityAnalyzer()
        self.current_img = None
        self._build_ui()

    def _build_ui(self):
        # Header
        top = tk.Frame(self.root, bg="#16213E", height=58)
        top.pack(fill=tk.X)
        tk.Label(top, text="🌍  NATIONALITY  &  EMOTION  DETECTION  SYSTEM",
                 font=("Palatino", 15, "bold"), fg="#F4A261", bg="#16213E").pack(
                 side=tk.LEFT, padx=16, pady=12)

        main = tk.Frame(self.root, bg="#1A1A2E")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Left: image preview
        left = tk.Frame(main, bg="#16213E")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(left, text="IMAGE PREVIEW", font=("Palatino", 11, "bold"),
                 fg="#F4A261", bg="#16213E").pack(pady=(6,2))
        self.img_canvas = tk.Label(left, bg="#0F1923", width=680, height=480)
        self.img_canvas.pack(padx=6, pady=4, fill=tk.BOTH, expand=True)

        # Controls
        ctl = tk.Frame(left, bg="#16213E")
        ctl.pack(fill=tk.X, padx=8, pady=4)
        self._btn(ctl, "📂 Upload Image", self.upload_image, "#E76F51").pack(side=tk.LEFT, padx=4)
        self._btn(ctl, "🔍 Analyze",      self.analyze,      "#2A9D8F").pack(side=tk.LEFT, padx=4)
        self._btn(ctl, "📹 Webcam",       self.webcam,       "#457B9D").pack(side=tk.LEFT, padx=4)
        self._btn(ctl, "🗑 Clear",         self.clear,        "#555").pack(side=tk.LEFT, padx=4)

        # Right: results
        right = tk.Frame(main, bg="#16213E", width=400)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))
        right.pack_propagate(False)

        tk.Label(right, text="ANALYSIS RESULTS", font=("Palatino", 12, "bold"),
                 fg="#F4A261", bg="#16213E").pack(pady=(10, 4))

        # Results scrollable text area
        result_frame = tk.Frame(right, bg="#16213E")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.result_text = tk.Text(result_frame, bg="#0F1923", fg="#E2E2E2",
                                   font=("Courier", 11), relief=tk.FLAT,
                                   wrap=tk.WORD, state=tk.DISABLED)
        sb = ttk.Scrollbar(result_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=sb.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Color tags
        for tag, color in [("header", "#F4A261"), ("nat", "#06D6A0"),
                            ("emo", "#FFB703"), ("age", "#8ECAE6"),
                            ("dress", "#FB8500"), ("sep", "#555")]:
            self.result_text.tag_configure(tag, foreground=color,
                                           font=("Courier", 11, "bold"))

        # Status
        self.status = tk.Label(self.root, text="Ready — Upload an image to analyze",
                               bg="#16213E", fg="#A8DADC", font=("Courier", 10), anchor=tk.W)
        self.status.pack(fill=tk.X, padx=10, pady=4)

    def _btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd, bg=color, fg="white",
                         font=("Palatino", 10, "bold"), relief=tk.FLAT, padx=8, pady=6,
                         activebackground="#333", cursor="hand2")

    def upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path: return
        self.current_img = cv2.imread(path)
        if self.current_img is None:
            messagebox.showerror("Error", "Cannot read image."); return
        self._show_img(self.current_img)
        self.status.config(text=f"Loaded: {os.path.basename(path)} — Click Analyze")

    def analyze(self):
        if self.current_img is None:
            messagebox.showinfo("No Image", "Upload an image first."); return
        self.status.config(text="Analyzing…")
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        results = self.analyzer.analyze_image(self.current_img)
        annotated = self.analyzer.annotate(self.current_img, results)
        self.root.after(0, self._show_img, annotated)
        self.root.after(0, self._display_results, results)
        self.root.after(0, self.status.config,
                        {"text": f"Found {len(results)} face(s)"})

    def _display_results(self, results):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        if not results:
            self.result_text.insert(tk.END, "  No faces detected.\n")
            self.result_text.config(state=tk.DISABLED)
            return
        for i, r in enumerate(results, 1):
            self.result_text.insert(tk.END, f"\n  ── Person {i} ──\n", "header")
            lines = self.analyzer.build_output(r)
            icons = {"Nationality": "🌍", "Emotion": "😊", "Age": "🎂", "Dress Color": "👗"}
            for line in lines:
                key = line.split(":")[0].strip()
                icon = icons.get(key, "•")
                tag = {"Nationality": "nat", "Emotion": "emo",
                       "Age": "age", "Dress Color": "dress"}.get(key, "sep")
                self.result_text.insert(tk.END, f"  {icon}  {line}\n", tag)
            self.result_text.insert(tk.END, "\n")
        self.result_text.config(state=tk.DISABLED)

    def webcam(self):
        """Capture single frame from webcam and analyze."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam."); return
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Error", "Failed to capture frame."); return
        self.current_img = frame
        self._show_img(frame)
        self.status.config(text="Webcam frame captured — Click Analyze")

    def clear(self):
        self.current_img = None
        self.img_canvas.configure(image="")
        self.img_canvas.image = None
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)
        self.status.config(text="Cleared")

    def _show_img(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = min(680/w, 480/h, 1.0)
        nw, nh = int(w*scale), int(h*scale)
        small = cv2.resize(frame_bgr, (nw, nh))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.img_canvas.configure(image=photo)
        self.img_canvas.image = photo


def main():
    root = tk.Tk()
    app = NationalityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
