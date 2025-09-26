import os
import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from backend.video_data import LipVideoDataset, collate_pad_batch
from backend.model import LipReadingNet


@dataclass
class TrainingConfig:
	epochs: int = 3
	batch_size: int = 4
	learning_rate: float = 1e-3
	val_split: float = 0.2
	max_samples_per_class: Optional[int] = None


def set_seed(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def _augment(frames: torch.Tensor) -> torch.Tensor:
	# frames: B,T,1,H,W in [0,1]
	if random.random() < 0.5:
		frames = frames + (torch.randn_like(frames) * 0.02)
	frames = torch.clamp(frames, 0.0, 1.0)
	return frames


def train_model(models_dir: str, data_dir: str, label_to_index: Dict[str, int], config: TrainingConfig) -> Dict:
	os.makedirs(models_dir, exist_ok=True)
	# Prefer Apple MPS on macOS, then CUDA, else CPU
	if torch.backends.mps.is_available():
		device = torch.device("mps")
	elif torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")
	dataset = LipVideoDataset(
		videos_root=data_dir,
		label_to_index=label_to_index,
		max_samples_per_class=config.max_samples_per_class,
	)
	if len(dataset) < 2:
		return {"message": "Not enough data to train"}
	val_size = max(1, int(len(dataset) * config.val_split))
	train_size = len(dataset) - val_size
	train_ds, val_ds = random_split(dataset, [train_size, val_size])

	train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_pad_batch)
	val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_pad_batch)

	model = LipReadingNet(num_classes=len(label_to_index))
	model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
	scheduler = CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))

	best_val = 0.0
	for epoch in range(config.epochs):
		model.train()
		for frames, lengths, labels in train_loader:
			frames = frames.to(device)
			frames = _augment(frames)
			labels = labels.to(device)
			optimizer.zero_grad()
			logits = model(frames, lengths)
			loss = criterion(logits, labels)
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
			optimizer.step()

		model.eval()
		correct = 0
		total = 0
		with torch.no_grad():
			for frames, lengths, labels in val_loader:
				frames = frames.to(device)
				labels = labels.to(device)
				logits = model(frames, lengths)
				preds = logits.argmax(dim=1)
				correct += (preds == labels).sum().item()
				total += labels.size(0)
		val_acc = correct / max(1, total)
		scheduler.step()

		torch.save({
			"state_dict": model.state_dict(),
			"label_to_index": label_to_index,
		}, os.path.join(models_dir, "model.pt"))

	return {"val_accuracy": val_acc}


def evaluate_accuracy(model: LipReadingNet, data_dir: str, label_to_index: Dict[str, int]) -> float:
	if torch.backends.mps.is_available():
		device = torch.device("mps")
	elif torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")
	dataset = LipVideoDataset(videos_root=data_dir, label_to_index=label_to_index)
	if len(dataset) == 0:
		return 0.0
	loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_pad_batch)
	model.to(device)
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for frames, lengths, labels in loader:
			frames = frames.to(device)
			labels = labels.to(device)
			logits = model(frames, lengths)
			preds = logits.argmax(dim=1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)
	return correct / max(1, total)
