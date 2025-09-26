import os
import sys
import threading
import time
import tempfile
from typing import Dict, List, Tuple
from collections import deque

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import simpledialog

from backend.inference import load_or_init_model, predict_video, reset_weights
from backend.training import train_model, TrainingConfig, evaluate_accuracy
from backend.video_data import build_label_mapping
from backend.self_learn import incremental_learn

DATA_DIR = os.path.join(ROOT, "videos")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


class LipGUI:
	def __init__(self, root: tk.Tk):
		self.root = root
		self.root.title("Lip Reading - Desktop")
		self.root.geometry("960x720")

		self.video_label = tk.Label(self.root)
		self.video_label.pack(pady=8)

		controls = tk.Frame(self.root)
		controls.pack(fill=tk.X, padx=8, pady=4)

		self.pred_var = tk.StringVar(value="Predict: …")
		pred_lbl = tk.Label(controls, textvariable=self.pred_var, font=("Helvetica", 16, "bold"))
		pred_lbl.pack(side=tk.LEFT, padx=8)

		btn_frame = tk.Frame(self.root)
		btn_frame.pack(fill=tk.X, padx=8, pady=8)

		self.start_btn = ttk.Button(btn_frame, text="Start Camera", command=self.toggle_camera)
		self.start_btn.pack(side=tk.LEFT, padx=6)

		self.predict_live = tk.BooleanVar(value=True)
		ttk.Checkbutton(btn_frame, text="Live Predict", variable=self.predict_live).pack(side=tk.LEFT)

		ttk.Button(btn_frame, text="Predict Now", command=self.on_predict_now).pack(side=tk.LEFT, padx=6)
		ttk.Button(btn_frame, text="Train", command=self.on_train).pack(side=tk.LEFT, padx=6)
		ttk.Button(btn_frame, text="Accuracy", command=self.on_accuracy).pack(side=tk.LEFT, padx=6)
		ttk.Button(btn_frame, text="Self-Learn", command=self.on_self_learn).pack(side=tk.LEFT, padx=6)
		ttk.Button(btn_frame, text="Reset", command=self.on_reset).pack(side=tk.LEFT, padx=6)
		ttk.Button(btn_frame, text="Help", command=self.on_help).pack(side=tk.LEFT, padx=6)
		ttk.Button(btn_frame, text="Quit", command=self.root.destroy).pack(side=tk.RIGHT, padx=6)

		self.cap = None
		self.running = False
		self.last_pred = ""
		self.pred_lock = threading.Lock()
		self.model = None
		self.label_to_index: Dict[str, int] = {}
		self.load_model()

		# Haar cascades for face and mouth
		self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
		self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
		self.prev_gray = None
		self.motion_cooldown_until = 0.0
		# Rolling buffer for continuous live prediction
		self.frame_buffer: deque[np.ndarray] = deque(maxlen=20)
		self.next_auto_predict_at: float = 0.0

	def load_model(self):
		self.model, self.label_to_index = load_or_init_model(MODELS_DIR, DATA_DIR)

	def toggle_camera(self):
		if self.running:
			self.running = False
			self.start_btn.config(text="Start Camera")
			if self.cap:
				self.cap.release()
				self.cap = None
			return
		self.cap = cv2.VideoCapture(0)
		if not self.cap.isOpened():
			messagebox.showerror("Camera", "Cannot open camera")
			return
		self.running = True
		self.start_btn.config(text="Stop Camera")
		threading.Thread(target=self.loop, daemon=True).start()

	def detect_mouth_roi(self, frame: np.ndarray) -> Tuple[int,int,int,int]:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
		if len(faces) == 0:
			# fallback lower-half
			h, w = gray.shape
			return 0, h//2, w, h//2
		x, y, w, h = faces[0]
		roi_gray = gray[y:y+h, x:x+w]
		mouths = self.mouth_cascade.detectMultiScale(roi_gray, 1.5, 20)
		if len(mouths) > 0:
			mx, my, mw, mh = mouths[0]
			# Ensure minimum size; else fallback to bottom-third
			if mw >= 40 and mh >= 30:
				return x+mx, y+my, mw, mh
		# bottom-third of face
		return x, y + (2*h)//3, w, h//3

	def loop(self):
		last_flush = time.time()
		while self.running and self.cap and self.cap.isOpened():
			ret, frame = self.cap.read()
			if not ret:
				continue
			# Mouth ROI with fallback to full frame when invalid
			x,y,w,h = self.detect_mouth_roi(frame)
			if w <= 0 or h <= 0:
				mouth = frame
				x,y,w,h = 0,0,frame.shape[1], frame.shape[0]
			else:
				mouth = frame[y:y+h, x:x+w]
			# Motion drawing using optical flow on mouth region
			gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
			if gray.size > 0:
				if self.prev_gray is None or self.prev_gray.shape != gray.shape:
					self.prev_gray = gray
				else:
					flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 1, 15, 3, 5, 1.2, 0)
					step = 12
					hh, ww = gray.shape
					ys, xs = np.mgrid[step//2:hh:step, step//2:ww:step]
					fx, fy = flow[ys, xs].transpose(2,0,1)
					lines = np.dstack((xs, ys, xs+fx, ys+fy)).reshape(-1,4).astype(np.int32)
					for (x1,y1,x2,y2) in lines:
						cv2.line(mouth, (x1,y1), (x2,y2), (0,255,0), 1)
					self.prev_gray = gray
			# Draw mouth box on original frame
			cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

			# Show preview
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			img = cv2.resize(rgb, (900, 520))
			from PIL import Image, ImageTk
			im = Image.fromarray(img)
			tk_img = ImageTk.PhotoImage(image=im)
			self.video_label.configure(image=tk_img)
			self.video_label.image = tk_img

			# Collect frames from cropped mouth for prediction
			if self.predict_live.get():
				mouth_resized = cv2.resize(mouth, (224, 224), interpolation=cv2.INTER_AREA)
				self.frame_buffer.append(mouth_resized.copy())
				now = time.time()
				# Auto predict every ~0.6s if we have enough frames
				if now >= self.next_auto_predict_at and len(self.frame_buffer) >= 8:
					clip = list(self.frame_buffer)[-12:]
					self.next_auto_predict_at = now + 0.6
					threading.Thread(target=self.predict_clip, args=(clip,), daemon=True).start()
			time.sleep(0.02)

	def predict_clip(self, frames: List[np.ndarray]):
		try:
			h, w = frames[0].shape[:2]
			fd, tmp_path = tempfile.mkstemp(suffix='.mp4')
			os.close(fd)
			fourcc = cv2.VideoWriter_fourcc(*'mp4v')
			vw = cv2.VideoWriter(tmp_path, fourcc, 15.0, (w, h))
			for fr in frames:
				vw.write(fr)
			vw.release()
			with open(tmp_path, 'rb') as v:
				content = v.read()
			os.remove(tmp_path)
			pred, probs = predict_video(self.model, content, self.label_to_index)
			self.root.after(0, lambda: self.pred_var.set(f"Predict: {pred}"))
		except Exception:
			self.root.after(0, lambda: self.pred_var.set(f"Predict: (error)"))

	def on_predict_now(self):
		# Force prediction from the latest frames in the buffer
		if len(self.frame_buffer) == 0:
			return
		clip = list(self.frame_buffer)[-12:]
		threading.Thread(target=self.predict_clip, args=(clip,), daemon=True).start()

	def on_train(self):
		label_to_index = build_label_mapping(DATA_DIR) or {"unknown": 0}
		cfg = TrainingConfig(epochs=3, batch_size=4, learning_rate=1e-3, val_split=0.2, max_samples_per_class=None)
		def _train():
			metrics = train_model(MODELS_DIR, DATA_DIR, label_to_index, cfg)
			self.load_model()
			messagebox.showinfo("Train", f"Done: {metrics}")
		threading.Thread(target=_train, daemon=True).start()

	def on_accuracy(self):
		def _acc():
			acc = evaluate_accuracy(self.model, DATA_DIR, self.label_to_index)
			messagebox.showinfo("Accuracy", f"Accuracy: {acc:.3f}")
		threading.Thread(target=_acc, daemon=True).start()

	def on_self_learn(self):
		label = simpledialog.askstring("Self-Learn", "Enter label for this clip:", parent=self.root) or "unknown"
		path = filedialog.askopenfilename(title="Select video for self-learn", filetypes=[("Video", "*.mp4 *.avi *.mov *.mpg *.mpeg *.mkv"), ("All", "*.*")])
		if not path:
			return
		def _sl():
			with open(path, 'rb') as f:
				metrics = incremental_learn(self.model, f.read(), label, MODELS_DIR, self.label_to_index)
			self.load_model()
			messagebox.showinfo("Self-Learn", f"Done: {metrics}")
		threading.Thread(target=_sl, daemon=True).start()

	def on_reset(self):
		reset_weights(MODELS_DIR)
		self.load_model()
		self.pred_var.set("Predict: …")
		messagebox.showinfo("Reset", "Model weights reset.")

	def on_help(self):
		msg = (
			"Desktop controls:\n"
			"- Start Camera: webcam preview and live mouth ROI prediction\n"
			"- Shows green motion vectors over the mouth region\n"
			"- Train: train on videos/ (label folders)\n"
			"- Accuracy: evaluate on videos/\n"
			"- Live Predict: shows word as soon as lip motion is detected\n"
			"- Self-Learn: prompts for a label, then fine-tunes with one video\n"
			"- Reset: clear saved weights\n"
		)
		messagebox.showinfo("Help", msg)


def main():
	root = tk.Tk()
	app = LipGUI(root)
	root.mainloop()


if __name__ == "__main__":
	main()
