import os
import sys
import argparse
import json

# Ensure project root is on sys.path when running as a script
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from backend.training import train_model, TrainingConfig, evaluate_accuracy
from backend.inference import load_or_init_model, predict_video, reset_weights
from backend.video_data import build_label_mapping
from backend.self_learn import incremental_learn

DATA_DIR = os.path.join(ROOT, "videos")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def cmd_train(args):
	label_to_index = build_label_mapping(DATA_DIR) or {"unknown": 0}
	cfg = TrainingConfig(
		epochs=args.epochs,
		batch_size=args.batch_size,
		learning_rate=args.lr,
		val_split=args.val_split,
		max_samples_per_class=args.max_samples_per_class,
	)
	metrics = train_model(MODELS_DIR, DATA_DIR, label_to_index, cfg)
	print(json.dumps(metrics))


def cmd_accuracy(args):
	model, label_to_index = load_or_init_model(MODELS_DIR, DATA_DIR)
	acc = evaluate_accuracy(model, DATA_DIR, label_to_index)
	print(json.dumps({"accuracy": acc}))


def cmd_predict(args):
	model, label_to_index = load_or_init_model(MODELS_DIR, DATA_DIR)
	with open(args.file, "rb") as f:
		pred, probs = predict_video(model, f.read(), label_to_index)
	print(json.dumps({"prediction": pred, "probabilities": probs}))


def cmd_self_learn(args):
	model, label_to_index = load_or_init_model(MODELS_DIR, DATA_DIR)
	with open(args.file, "rb") as f:
		metrics = incremental_learn(model, f.read(), args.label, MODELS_DIR, label_to_index)
	print(json.dumps(metrics))


def cmd_reset(args):
	reset_weights(MODELS_DIR)
	print(json.dumps({"status": "reset"}))


def main():
	p = argparse.ArgumentParser(description="Lip Reading CLI")
	sub = p.add_subparsers(dest="cmd", required=True)
	pt = sub.add_parser("train")
	pt.add_argument("--epochs", type=int, default=3)
	pt.add_argument("--batch-size", type=int, default=4)
	pt.add_argument("--lr", type=float, default=1e-3)
	pt.add_argument("--val-split", type=float, default=0.2)
	pt.add_argument("--max-samples-per-class", type=int, default=None)
	pt.set_defaults(func=cmd_train)

	pa = sub.add_parser("accuracy")
	pa.set_defaults(func=cmd_accuracy)

	pp = sub.add_parser("predict")
	pp.add_argument("file", help="Path to video file")
	pp.set_defaults(func=cmd_predict)

	ps = sub.add_parser("self-learn")
	ps.add_argument("label", help="Label name")
	ps.add_argument("file", help="Path to video file")
	ps.set_defaults(func=cmd_self_learn)

	pr = sub.add_parser("reset")
	pr.set_defaults(func=cmd_reset)

	args = p.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
