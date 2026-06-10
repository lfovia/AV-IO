import os
import cv2
import torch
import librosa
import numpy as np
from moviepy.editor import VideoFileClip
from transformers import (
    CLIPProcessor,
    CLIPModel,
    ClapProcessor,
    ClapModel,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

############################################################
# LOAD MODELS
############################################################

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(DEVICE)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

clap_model = ClapModel.from_pretrained(
    "laion/clap-htsat-unfused"
).to(DEVICE)

clap_processor = ClapProcessor.from_pretrained(
    "laion/clap-htsat-unfused"
)

############################################################
# TEXT PROMPTS
############################################################

PROMPTS = [
    "a photo of an indoor scene",
    "a photo of an outdoor scene"
]

############################################################
# VIDEO PROCESSING
############################################################

def extract_center_frame(video_path):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    center_idx = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, center_idx)

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError("Could not read center frame.")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame


def extract_audio(video_path, sr=16000, duration=10):

    tmp_audio = "temp.wav"

    try:
        video = VideoFileClip(video_path)

        if video.audio is None:
            return np.zeros(sr * duration)

        video.audio.write_audiofile(
            tmp_audio,
            fps=sr,
            logger=None
        )

        audio, _ = librosa.load(
            tmp_audio,
            sr=sr
        )

        target_len = sr * duration

        if len(audio) < target_len:
            padded = np.zeros(target_len)
            padded[:len(audio)] = audio
            audio = padded
        else:
            audio = audio[:target_len]

        os.remove(tmp_audio)

        return audio

    except Exception:

        if os.path.exists(tmp_audio):
            os.remove(tmp_audio)

        return np.zeros(sr * duration)

############################################################
# FEATURE EXTRACTION
############################################################

@torch.no_grad()
def encode_clip_image(image):

    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    ).to(DEVICE)

    feat = clip_model.get_image_features(**inputs)

    feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat


@torch.no_grad()
def encode_clip_text(texts):

    inputs = clip_processor(
        text=texts,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    feat = clip_model.get_text_features(**inputs)

    feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat


@torch.no_grad()
def encode_clap_audio(audio):

    inputs = clap_processor(
        audios=audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).to(DEVICE)

    feat = clap_model.get_audio_features(**inputs)

    feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat


@torch.no_grad()
def encode_clap_text(texts):

    inputs = clap_processor(
        text=texts,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    feat = clap_model.get_text_features(**inputs)

    feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat

############################################################
# SIMILARITY
############################################################

def cosine_similarity(a, b):

    return torch.matmul(a, b.T)


############################################################
# MAJORITY VOTING
############################################################

def majority_vote(indoor_score, outdoor_score):

    if indoor_score > outdoor_score:
        pred = "indoor"
    else:
        pred = "outdoor"

    confidence = abs(indoor_score - outdoor_score)

    return pred, confidence


############################################################
# FULL PIPELINE
############################################################

def classify_video(video_path):

    frame = extract_center_frame(video_path)

    audio = extract_audio(video_path)

    ########################################################
    # CLIP
    ########################################################

    image_feat = encode_clip_image(frame)

    clip_text_feat = encode_clip_text(PROMPTS)

    clip_scores = cosine_similarity(
        image_feat,
        clip_text_feat
    )[0]

    clip_indoor = clip_scores[0].item()
    clip_outdoor = clip_scores[1].item()

    clip_pred, clip_conf = majority_vote(
        clip_indoor,
        clip_outdoor
    )

    ########################################################
    # CLAP
    ########################################################

    audio_feat = encode_clap_audio(audio)

    clap_text_feat = encode_clap_text(PROMPTS)

    clap_scores = cosine_similarity(
        audio_feat,
        clap_text_feat
    )[0]

    clap_indoor = clap_scores[0].item()
    clap_outdoor = clap_scores[1].item()

    clap_pred, clap_conf = majority_vote(
        clap_indoor,
        clap_outdoor
    )

    ########################################################
    # FINAL DECISION
    ########################################################

    if clip_conf >= clap_conf:
        final_pred = clip_pred
        source = "CLIP"
    else:
        final_pred = clap_pred
        source = "CLAP"

    return {
        "clip_prediction": clip_pred,
        "clip_confidence": clip_conf,
        "clap_prediction": clap_pred,
        "clap_confidence": clap_conf,
        "final_prediction": final_pred,
        "selected_modality": source,
    }


############################################################
# TEST
############################################################

if __name__ == "__main__":

    result = classify_video("sample_video.mp4")

    print(result)