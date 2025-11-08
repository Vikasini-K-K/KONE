# lift_fault_detection.py

import os
from collections import Counter

import moviepy.editor as mp
from PIL import Image

# --- Visual classification ---
import torch
from transformers import CLIPProcessor, CLIPModel

# --- Audio classification ---
import tensorflow as tf
import tensorflow_hub as hub
import librosa

# -----------------------------
# USER INPUT: video path
video_path = "video2.mp4"  # <-- Replace with your video file
frame_folder = "frames"

os.makedirs(frame_folder, exist_ok=True)

# -----------------------------
# STEP 1: Extract frames + audio
print("Extracting frames and audio...")
video = mp.VideoFileClip(video_path)
audio_path = "video_audio.wav"
video.audio.write_audiofile(audio_path)
video.write_images_sequence(f"{frame_folder}/frame_%04d.jpg", fps=1)  # 1 frame per second
print("Frames and audio extracted!")

# -----------------------------
# STEP 2: Visual classification with CLIP
print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

classes = ["elevator motor", "door mechanism", "pulley and ropes",
           "control panel", "cabin interior", "sensor area"]

def classify_frame(image_path):
    image = Image.open(image_path)
    inputs = clip_processor(text=classes, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return classes[probs.argmax()]

print("Classifying frames...")
frame_predictions = []
for filename in sorted(os.listdir(frame_folder)):
    if filename.endswith(".jpg"):
        pred = classify_frame(os.path.join(frame_folder, filename))
        frame_predictions.append(pred)

most_common_part = Counter(frame_predictions).most_common(1)[0][0]
print("Most detected lift part:", most_common_part)

# -----------------------------
# STEP 3: Audio classification with YAMNet (updated to show top predictions)
print("Loading YAMNet model...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load audio
waveform, sr = librosa.load(audio_path, sr=16000)
waveform = waveform.astype('float32')

scores, embeddings, spectrogram = yamnet_model(waveform)
class_map_path = yamnet_model.class_map_path().numpy()
with open(class_map_path, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Get top N audio predictions
mean_scores = tf.reduce_mean(scores, axis=0).numpy()
top_n = 5  # number of top classes to display
top_indices = mean_scores.argsort()[-top_n:][::-1]

print("Detected sounds (top classes with confidence):")
for idx in top_indices:
    print(f"  {class_names[idx]}: {mean_scores[idx]:.3f}")

# Use top prediction for fault logic
sound_label = class_names[top_indices[0]]

# -----------------------------
# STEP 4: Fault prediction (simple logic)
fault = "Abnormal behavior observed; manual check of the detected part recommended"

if "motor" in most_common_part.lower() and any(k in sound_label.lower() for k in ["grinding", "engine", "motor"]):
    fault = "Bearing wear or misalignment in motor"
elif "door" in most_common_part.lower() and "squeak" in sound_label.lower():
    fault = "Door track friction or lubrication issue"
elif "pulley" in most_common_part.lower() and "metal" in sound_label.lower():
    fault = "Pulley or rope wear"

# -----------------------------
# STEP 5: Generate prompt
prompt = f"""
Detected part: {most_common_part}
Detected sound: {sound_label}
Predicted fault: {fault}

Recommendation: Check the {most_common_part} assembly for {sound_label} issues and apply necessary maintenance.
"""

print("\n=== FINAL FAULT REPORT ===")
print(prompt)
