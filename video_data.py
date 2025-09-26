import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import tempfile


SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mpg", ".mpeg", ".mkv"}


def list_class_videos(videos_root: str) -> Dict[str, List[str]]:
	label_to_paths: Dict[str, List[str]] = {}
	if not os.path.isdir(videos_root):
		return label_to_paths
	# If videos are directly under root, group into a single 'unknown' label
	root_files = [f for f in os.listdir(videos_root) if os.path.isfile(os.path.join(videos_root, f))]
	video_files = [os.path.join(videos_root, f) for f in root_files if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
	if video_files:
		label_to_paths["unknown"] = sorted(video_files)
	# Also support labeled subdirectories
	for entry in os.listdir(videos_root):
		label_dir = os.path.join(videos_root, entry)
		if not os.path.isdir(label_dir):
			continue
		paths: List[str] = []
		for fname in os.listdir(label_dir):
			ext = os.path.splitext(fname)[1].lower()
			if ext in SUPPORTED_EXTS:
				paths.append(os.path.join(label_dir, fname))
		if paths:
			label_to_paths[entry] = sorted(paths)
	return label_to_paths


def build_label_mapping(videos_root: str) -> Dict[str, int]:
	label_to_paths = list_class_videos(videos_root)
	labels = sorted(label_to_paths.keys())
	return {label: i for i, label in enumerate(labels)}


def read_video_frames(path_or_bytes: bytes | str, max_frames: int = 64, target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
	# OpenCV cannot reliably read from raw bytes; if bytes provided, write a unique temp file
	if isinstance(path_or_bytes, bytes):
		with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
			tmp.write(path_or_bytes)
			tmp_path = tmp.name
		cap = cv2.VideoCapture(tmp_path)
	else:
		cap = cv2.VideoCapture(path_or_bytes)
	frames: List[np.ndarray] = []
	count = 0
	while count < max_frames:
		ret, frame = cap.read()
		if not ret:
			break
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		h, w = frame.shape
		min_dim = min(h, w)
		start_y = (h - min_dim) // 2
		start_x = (w - min_dim) // 2
		crop = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]
		crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
		frames.append(crop)
		count += 1
	cap.release()
	# Clean up temp file if we created one
	if isinstance(path_or_bytes, bytes):
		try:
			os.remove(tmp_path)
		except Exception:
			pass
	if len(frames) == 0:
		return np.zeros((1, *target_size), dtype=np.uint8)
	return np.stack(frames, axis=0)


class LipVideoDataset(Dataset):
	def __init__(self, videos_root: str, label_to_index: Dict[str, int], max_samples_per_class: Optional[int] = None):
		self.items: List[Tuple[str, int]] = []
		for label, idx in label_to_index.items():
			label_dir = os.path.join(videos_root, label)
			paths: List[str] = []
			if os.path.isdir(label_dir):
				paths = [os.path.join(label_dir, f) for f in os.listdir(label_dir)
						 if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
			else:
				# Possibly unlabeled root case 'unknown'
				if label == "unknown":
					paths = [os.path.join(videos_root, f) for f in os.listdir(videos_root)
							 if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
			paths = sorted(paths)
			if max_samples_per_class is not None:
				paths = paths[:max_samples_per_class]
			for p in paths:
				self.items.append((p, idx))

	def __len__(self) -> int:
		return len(self.items)

	def __getitem__(self, index: int):
		path, label = self.items[index]
		frames = read_video_frames(path)  # T, H, W
		frames = frames.astype(np.float32) / 255.0
		frames = np.expand_dims(frames, 1)  # T, 1, H, W
		length = frames.shape[0]
		T = 64
		if length < T:
			pad = np.zeros((T - length, 1, frames.shape[2], frames.shape[3]), dtype=frames.dtype)
			frames = np.concatenate([frames, pad], axis=0)
		else:
			frames = frames[:T]
		return torch.from_numpy(frames), length, torch.tensor(label, dtype=torch.long)


def collate_pad_batch(batch):
	frames, lengths, labels = zip(*batch)
	frames = torch.stack(frames, dim=0)
	lengths = torch.tensor(lengths, dtype=torch.long)
	labels = torch.stack(labels, dim=0)
	return frames, lengths, labels
