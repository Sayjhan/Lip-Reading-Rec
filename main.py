import os
import asyncio
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.training import train_model, TrainingConfig, evaluate_accuracy
from backend.inference import predict_video, load_or_init_model, reset_weights
from backend.video_data import build_label_mapping
from backend.self_learn import incremental_learn

APP_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(APP_DIR)
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
ASSETS_DIR = os.path.join(FRONTEND_DIR, "assets")
MODELS_DIR = os.path.join(APP_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "videos")

app = FastAPI(title="Lip Reading Service")

if os.path.isdir(FRONTEND_DIR):
	if os.path.isdir(ASSETS_DIR):
		app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


class TrainRequest(BaseModel):
	epochs: int = 3
	batch_size: int = 4
	learning_rate: float = 1e-3
	val_split: float = 0.2
	max_samples_per_class: Optional[int] = None


@app.get("/")
async def root():
	index_path = os.path.join(FRONTEND_DIR, "index.html")
	if os.path.exists(index_path):
		with open(index_path, "r") as f:
			return HTMLResponse(f.read())
	return JSONResponse({"message": "Lip Reading Service running"})


@app.get("/help")
async def help_endpoint():
	return JSONResponse({
		"endpoints": {
			"/train": "POST: Train model with TrainingConfig",
			"/predict": "POST: Upload video file for prediction",
			"/predict_frames": "POST: Upload short clip frames (multipart) for realtime",
			"/accuracy": "GET: Evaluate accuracy on dataset",
			"/self_learn": "POST: Upload labeled video to fine-tune",
			"/reset": "POST: Reset model weights",
			"/quit": "POST: Stop the server"
		}
	})


@app.post("/train")
async def train_endpoint(cfg: TrainRequest):
	label_to_index = build_label_mapping(DATA_DIR)
	if not label_to_index:
		label_to_index = {"unknown": 0}
	metrics = train_model(
		models_dir=MODELS_DIR,
		data_dir=DATA_DIR,
		label_to_index=label_to_index,
		config=TrainingConfig(
			epochs=cfg.epochs,
			batch_size=cfg.batch_size,
			learning_rate=cfg.learning_rate,
			val_split=cfg.val_split,
			max_samples_per_class=cfg.max_samples_per_class,
		),
	)
	return JSONResponse({"status": "ok", "metrics": metrics})


@app.get("/accuracy")
async def accuracy_endpoint():
	model, label_to_index = load_or_init_model(MODELS_DIR, DATA_DIR)
	acc = evaluate_accuracy(model, DATA_DIR, label_to_index)
	return JSONResponse({"accuracy": acc})


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
	model, label_to_index = load_or_init_model(MODELS_DIR, DATA_DIR)
	content = await file.read()
	pred, probs = predict_video(model, content, label_to_index)
	return JSONResponse({"prediction": pred, "probabilities": probs})


@app.post("/predict_frames")
async def predict_frames_endpoint(files: List[UploadFile] = File(...)):
	# Expect a short sequence of frames, will create a temp mp4/webm to reuse pipeline
	import cv2, numpy as np, tempfile
	model, label_to_index = load_or_init_model(MODELS_DIR, DATA_DIR)
	frames = []
	for f in files:
		data = await f.read()
		arr = np.frombuffer(data, dtype=np.uint8)
		img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
		if img is None:
			continue
		frames.append(img)
	if not frames:
		return JSONResponse({"error": "no frames"}, status_code=400)
	# Write temp video with approximate 15 fps
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
	pred, probs = predict_video(model, content, label_to_index)
	return JSONResponse({"prediction": pred, "probabilities": probs})


@app.post("/self_learn")
async def self_learn_endpoint(label: str = Form(...), file: UploadFile = File(...)):
	model, label_to_index = load_or_init_model(MODELS_DIR, DATA_DIR)
	content = await file.read()
	metrics = incremental_learn(model, content, label, MODELS_DIR, label_to_index)
	return JSONResponse({"status": "ok", "metrics": metrics})


@app.post("/reset")
async def reset_endpoint():
	reset_weights(MODELS_DIR)
	return JSONResponse({"status": "reset"})


@app.post("/quit")
async def quit_endpoint():
	async def shutdown():
		await asyncio.sleep(0.5)
		os._exit(0)
	asyncio.create_task(shutdown())
	return JSONResponse({"status": "shutting_down"})


if __name__ == "__main__":
	import uvicorn
	uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
