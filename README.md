# 🎤 Real-Time Voice vs Noise Detection System

---

## 📌 Overview

This project presents a **Real-Time Voice vs Noise Detection System** using machine learning and audio signal processing. The system captures live audio input through a microphone and classifies it as either **human voice** or **environmental noise**.

It utilizes **MFCC (Mel Frequency Cepstral Coefficients)** for feature extraction and a **Feedforward Neural Network (MLP)** for classification.

---

## 🚀 Features

* 🎤 Real-time audio detection using microphone
* 🧠 Machine learning-based classification
* 📊 MFCC feature extraction
* ⚡ Fast and lightweight implementation
* 💻 Runs efficiently on CPU

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Librosa
* NumPy
* Scikit-learn
* SoundDevice

<p align="center">
  <img src="https://media.giphy.com/media/ZVik7pBtu9dNS/giphy.gif" width="400"/>
</p>

---

## 🧠 Model Details

* Model Type: Feedforward Neural Network (MLP)
* Input Features: 40 MFCC coefficients
* Hidden Layers: 128 → 64 neurons
* Activation: ReLU
* Output: Sigmoid (Binary Classification)

---

## 📁 Project Structure

```
Dataset/
   ├── voice/
   └── noise/

realtime_voice_detection.py
voice_noise_train.py
voice_noise_model.pth
voice_noise_scaler.save
```

---

## ⚙️ Installation

```
pip install torch librosa scikit-learn joblib sounddevice numpy
```

---

## ▶️ Usage

```
python voice_noise_train.py
python realtime_voice_detection.py
```

---

## 📊 Sample Output

```
System Ready 🎤
Listening...

HUMAN VOICE detected | Confidence: 0.973
NOISE detected | Confidence: 0.741
HUMAN VOICE detected | Confidence: 0.998
```

<p align="center">
  <img src="https://media.giphy.com/media/3o7aD2saalBwwftBIY/giphy.gif" width="450"/>
</p>

---

## ⚠️ Limitations

* Small dataset may reduce accuracy
* Sensitive to noisy environments
* Only binary classification

---

## 🔮 Future Scope

* Multi-class sound detection (gunshot, scream, etc.)
* CNN-based deep learning models
* Speech-to-text integration
* Deployment as a web/mobile app

---

## 🎯 Applications

* Voice assistants
* Smart surveillance
* Noise filtering systems
* AI-based audio monitoring

---

## 👨‍💻 Author

**Sounak Maiti**

---

## 📜 License

This project is developed for academic and educational purposes.

<p align="center">
  <img src="https://media.giphy.com/media/26AHONQ79FdWZhAI0/giphy.gif" width="400"/>
</p>
