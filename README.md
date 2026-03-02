# 🤖 Advanced Multimodal AI System

A transformer-based multimodal intelligence system that analyzes:

- 📝 Text sentiment (RoBERTa)
- 🖼 Image classification (Vision Transformer - ViT)
- 🎙 Speech recognition + tone analysis (Wav2Vec2)
- 🔎 Confidence-weighted multimodal fusion scoring

---

## 🚀 Overview

This project integrates multiple transformer models to perform real-time multimodal analysis.  
It processes text, image, and audio inputs independently and then combines the signals using a weighted fusion logic to generate a contextual interpretation.

---

## 🧠 Architecture

Text Input → Sentiment Model  
Image Input → Vision Transformer  
Audio Input → Speech-to-Text → Sentiment  
→ Fusion Layer → Final Intelligence Summary  

---

## 🛠 Tech Stack

- Python
- HuggingFace Transformers
- PyTorch
- Gradio

---

## 📦 Installation

```bash
pip install -r requirements.txt
python app.py
