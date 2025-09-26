import os
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from backend.model import LipReadingNet
from backend.video_data import read_video_frames, build_label_mapping


def load_or_init_model(models_dir: str, data_dir: str) -> Tuple[LipReadingNet, Dict[str, int]]:
	os.makedirs(models_dir, exist_ok=True)
	label_to_index = build_label_mapping(data_dir)
	ckpt_path = os.path.join(models_dir, "model.pt")
	if os.path.exists(ckpt_path):
		ckpt = torch.load(ckpt_path, map_location="cpu")
		if isinstance(ckpt, dict) and "label_to_index" in ckpt and ckpt["label_to_index"]:
			label_to_index = ckpt["label_to_index"]
		model = LipReadingNet(num_classes=len(label_to_index))
		state = ckpt.get("state_dict", ckpt)
		# Load non-strict to allow classifier shape changes
		missing, unexpected = model.load_state_dict(state, strict=False)
		# If classifier is mismatched, it will be left initialized with correct shape
		return model, label_to_index
	if not label_to_index:
		label_to_index = {"unknown": 0}
	model = LipReadingNet(num_classes=len(label_to_index))
	return model, label_to_index


def reset_weights(models_dir: str):
	ckpt_path = os.path.join(models_dir, "model.pt")
	if os.path.exists(ckpt_path):
		os.remove(ckpt_path)


def predict_video(model: LipReadingNet, video_bytes: bytes, label_to_index: Dict[str, int]):
	model.eval()
	frames = read_video_frames(video_bytes)  # T,H,W
	frames = frames.astype("float32") / 255.0
	frames = frames[:64]
	length = frames.shape[0]
	frames = frames[None, :, None, :, :]  # 1,T,1,H,W
	tensor = torch.from_numpy(frames)
	with torch.no_grad():
		logits = model(tensor, torch.tensor([length]))
		probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
	index_to_label = {v: k for k, v in label_to_index.items()}
	pred_idx = int(probs.argmax())
	pred_label = index_to_label.get(pred_idx, str(pred_idx))
	return pred_label, {index_to_label.get(i, str(i)): float(probs[i]) for i in range(len(probs))}
