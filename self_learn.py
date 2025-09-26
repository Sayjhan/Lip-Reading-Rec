import os
from typing import Dict

import torch
from torch import nn
from torch.optim import Adam

from backend.model import LipReadingNet
from backend.video_data import read_video_frames


def incremental_learn(model: LipReadingNet, video_bytes: bytes, label: str, models_dir: str, label_to_index: Dict[str, int]):
	model.train()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	# Ensure label exists
	if label not in label_to_index:
		label_to_index[label] = max(label_to_index.values(), default=-1) + 1
		# Expand classifier head if needed
		old_head = model.classifier
		in_features = old_head.in_features
		new_head = nn.Linear(in_features, len(label_to_index))
		with torch.no_grad():
			new_head.weight[:old_head.out_features].copy_(old_head.weight)
			new_head.bias[:old_head.out_features].copy_(old_head.bias)
		model.classifier = new_head.to(device)

	frames = read_video_frames(video_bytes)  # T,H,W
	frames = frames.astype("float32") / 255.0
	frames = frames[:64]
	length = frames.shape[0]
	frames = frames[None, :, None, :, :]  # 1,T,1,H,W
	tensor = torch.from_numpy(frames).to(device)
	label_idx = torch.tensor([label_to_index[label]], dtype=torch.long).to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = Adam(model.parameters(), lr=1e-4)

	loss_val = 0.0
	for _ in range(50):  # a few steps of fine-tune
		optimizer.zero_grad()
		logits = model(tensor, torch.tensor([length]).to(device))
		loss = criterion(logits, label_idx)
		loss.backward()
		optimizer.step()
		loss_val = float(loss.item())

	os.makedirs(models_dir, exist_ok=True)
	torch.save({
		"state_dict": model.state_dict(),
		"label_to_index": label_to_index,
	}, os.path.join(models_dir, "model.pt"))

	return {"loss": loss_val}
