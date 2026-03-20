import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib

VOICE_DIR = "Dataset/voice"
NOISE_DIR = "Dataset/noise"

MODEL_PATH = "voice_noise_model.pth"
SCALER_PATH = "voice_noise_scaler.save"

EPOCHS = 30


def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    return np.mean(mfcc.T, axis=0)


def load_dataset():

    X = []
    y = []

    for file in os.listdir(VOICE_DIR):

        path = os.path.join(VOICE_DIR, file)

        try:
            features = extract_features(path)
            X.append(features)
            y.append(1)
        except:
            pass

    for file in os.listdir(NOISE_DIR):

        path = os.path.join(NOISE_DIR, file)

        try:
            features = extract_features(path)
            X.append(features)
            y.append(0)
        except:
            pass

    return np.array(X), np.array(y)


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


print("Loading dataset...")

X,y = load_dataset()

print("Total samples:",len(X))

scaler = StandardScaler()

X = scaler.fit_transform(X)

X = torch.tensor(X,dtype=torch.float32)
y = torch.tensor(y,dtype=torch.float32).unsqueeze(1)

model = VoiceNoiseModel()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

print("Training started...\n")

for epoch in range(EPOCHS):

    optimizer.zero_grad()

    outputs = model(X)

    loss = criterion(outputs,y)

    loss.backward()

    optimizer.step()

    if (epoch+1)%5==0:

        print("Epoch",epoch+1,"Loss:",loss.item())

torch.save(model.state_dict(),MODEL_PATH)
joblib.dump(scaler,SCALER_PATH)

print("\nModel saved successfully!")