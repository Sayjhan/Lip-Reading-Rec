from typing import Tuple
import torch
from torch import nn


class TemporalConvBlock(nn.Module):
	def __init__(self, in_ch: int, mid_ch: int, out_ch: int, p: float = 0.3):
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
			nn.BatchNorm2d(mid_ch),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Dropout2d(p),
			nn.MaxPool2d(2),
		)

	def forward(self, x):  # x: B,T,C,H,W
		B, T, C, H, W = x.shape
		x = x.view(B * T, C, H, W)
		x = self.block(x)
		C2, H2, W2 = x.shape[1], x.shape[2], x.shape[3]
		x = x.view(B, T, C2, H2, W2)
		return x


class LipReadingNet(nn.Module):
	def __init__(self, num_classes: int):
		super().__init__()
		self.backbone1 = TemporalConvBlock(1, 32, 64, p=0.2)
		self.backbone2 = TemporalConvBlock(64, 64, 128, p=0.3)
		self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
		self.rnn = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
		self.classifier = nn.Sequential(
			nn.Dropout(0.3),
			nn.Linear(128 * 2, num_classes)
		)

	def forward(self, frames, lengths):  # frames: B,T,1,112,112 (or resized)
		x = self.backbone1(frames)
		x = self.backbone2(x)
		B, T, C, H, W = x.shape
		x = x.view(B * T, C, H, W)
		x = self.global_pool(x)
		x = x.view(B, T, C, 1, 1).squeeze(-1).squeeze(-1)  # B,T,C
		packed, _ = self.rnn(x)
		out = []
		for i in range(B):
			L = int(lengths[i].item())
			L = max(1, min(L, T))
			out.append(packed[i, L - 1, :])
		out = torch.stack(out, dim=0)
		logits = self.classifier(out)
		return logits
