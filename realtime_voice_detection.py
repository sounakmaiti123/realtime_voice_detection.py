import sounddevice as sd
import numpy as np
import librosa
import torch
import torch.nn as nn
import joblib

MODEL_PATH = "voice_noise_model.pth"
SCALER_PATH = "voice_noise_scaler.save"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE = 22050
DURATION = 2  # seconds


class VoiceNoiseModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(40,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,1),
            nn.Sigmoid()
        )

    def forward(self,x):

        return self.network(x)


def extract_features(audio):

    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)

    return np.mean(mfcc.T, axis=0)


print("Loading model...")

model = VoiceNoiseModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

scaler = joblib.load(SCALER_PATH)

print("System Ready 🎤")
print("Listening...\n")


while True:

    audio = sd.rec(int(SAMPLE_RATE * DURATION),
                   samplerate=SAMPLE_RATE,
                   channels=1)

    sd.wait()

    audio = audio.flatten()

    features = extract_features(audio)

    features = scaler.transform([features])

    features = torch.tensor(features, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():

        output = model(features)

        prob = output.item()

    if prob > 0.98:
        print("🗣 HUMAN VOICE detected | Confidence:", round(prob,3))
    else:
        print("🔊 NOISE detected | Confidence:", round(prob,3))