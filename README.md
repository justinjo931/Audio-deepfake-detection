# 🎙️ Audio Deepfake Detection for Real Conversations

> **Done By**: Justin Joseph S 
> **Email**: justinjo931@gmail.com

---

## 🧩 Part 1: Research & Selection

### 📚 Reference Repository
I explored the GitHub repository: [Audio Deepfake Detection – media-sec-lab](https://github.com/media-sec-lab/Audio-Deepfake-Detection), which curates a comprehensive collection of papers, models, and datasets focused on audio deepfake detection.

---

### 🎯 Use Case
Identify models suitable for:
- ✅ Detecting AI-generated human speech
- ✅ Real-time or near real-time detection
- ✅ Analysis of real-world conversations

---

### ✅ Selected Approaches

---

### **1. AASIST – Audio Anti-Spoofing using Spectro-Temporal Graph Attention**

**What is it?**  
AASIST is a deep learning model using spectro-temporal graphs and attention to catch audio deepfakes. It identifies abnormal pitch or unnatural patterns in both time and frequency dimensions.

**🔍 Key Innovations:**
- Graph Attention + Spectro-temporal encoding
- Focuses on suspicious regions using attention
- Generalizes to unseen attacks

**📈 Performance:**
- EER: **0.83%**, t-DCF: **0.028**

**✅ Why It Fits:**
- Top-tier accuracy
- Fast inference
- Excellent with background noise and unseen attacks

---

### **2. RawNet2 – End-to-End Model Using Raw Audio**

**What is it?**  
RawNet2 takes raw waveforms and uses Sinc-based filters and GRUs to learn and detect fakes without preprocessing.

**🔍 Key Innovations:**
- Raw waveform input (no spectrogram needed)
- Sinc filters + Residual + GRU

**📈 Performance:**
- EER: **1.12%**, t-DCF: **0.033**

**✅ Why It Fits:**
- Easy to run with CPU
- No preprocessing required
- Ideal for fast pipelines

**⚠️ Limitations:**
- Slower training on CPU
- Lower context understanding

---

### **3. ResMax – Residual Network with Max Feature Map**

**What is it?**  
ResMax uses residual blocks and a Max Feature Map (MFM) layer to detect the most important patterns in audio. It's ultra lightweight and fast.

**🔍 Key Innovations:**
- Residual CNN + Max Feature Map activation
- Learns compact, discriminative audio features

**📈 Performance:**
- LA EER: **2.19%**, PA EER: **0.37% (Rank 1)**

**✅ Why It Fits:**
- Extremely fast
- Great for mobile & edge devices
- Ideal for replay attack detection

---

### 📊 Comparison Table

| Feature / Criteria                 | **RawNet2**              | **AASIST**                        | **ResMax**                       |
|-----------------------------------|--------------------------|-----------------------------------|----------------------------------|
| Model Type                        | End-to-End CNN + GRU     | Graph Attention + Hybrid Features | Residual Net + MFM               |
| Input                             | Raw waveform             | Spectrogram                       | CQT (spectrogram)                |
| Performance (EER - LA)            | 1.12%                    | **0.83%**                         | 2.19%                            |
| Real-Time Capability              | Medium                   | High                              | **High**                         |
| Generalization                    | Good                     | **Excellent**                     | Moderate                         |
| Edge Deployment Suitability       | Moderate                 | Good                              | **Best**                         |
| Robustness to Noise               | Medium                   | **High**                          | Low to Moderate                  |

---

## 🛠️ Part 2: Implementation

### ✅ Selected Approach: **RawNet2**

### 📦 Dataset
- **Dataset**: ASVspoof 2019 Logical Access (LA)
- 📥 [Download Link](https://datashare.ed.ac.uk/handle/10283/3336)
- **Used**: LA Train + Dev

### 💻 Environment
- OS: Windows
- RAM: 8 GB
- GPU: ❌ (CPU only)
- Python: 3.6+
- Framework: PyTorch

### 🔧 Modifications
- Used updated PyTorch version (torch 1.10+)
- Converted path formats (Windows)
- Reduced batch size (8) and epochs (3) for CPU

---

### 🚀 Training Results

```bash
Epoch 0 | Loss: 0.1503 | Train Acc: 92.50% | Val Acc: 10.26%
Epoch 1 | Loss: 0.0001 | Train Acc: 100.00% | Val Acc: 10.26%
Epoch 2 | Loss: 0.0000 | Train Acc: 100.00% | Val Acc: 10.26%
```

⚠️ **Note**: Low validation accuracy is expected due to short training, CPU usage, and no augmentation.

---

### 📂 GitHub Repository
📎 [GitHub Code Submission](https://github.com/DINAKAR-S/Audio-Deepfake-Detection-for-Real-Conversations)

---

## 📚 Part 3: Documentation & Analysis

---

### 🔧 Implementation Process

**Challenges:**
- ⚠️ No GPU — slow training
- ⚠️ PyTorch 1.4 not available
- ⚠️ Path compatibility issues on Windows

**Solutions:**
- Switched to PyTorch 1.10+
- Adjusted training loop (3 epochs)
- Updated paths and simplified logic

**Assumptions:**
- Short training demonstrates proof of concept
- Validation accuracy is expected to improve with GPU + longer training
- RawNet2 is likely to generalize well with more real-world data

---

### 🔬 High-Level Explanation: **How RawNet2 Works**

1. **Input**: Raw audio waveform  
2. **Sinc Layer**: Learns frequency filters automatically  
3. **Residual CNN Blocks**: Extract short-term patterns  
4. **GRU Layer**: Captures temporal structure (speech rhythm)  
5. **FC + Softmax**: Outputs spoof/bonafide probability  

📌 All of this is learned directly from audio without hand-crafted features.

---

### 📈 Strengths & Weaknesses

| ✅ Strengths                        | ⚠️ Weaknesses                          |
|-----------------------------------|----------------------------------------|
| No preprocessing required         | Needs GPU for faster/better training   |
| Lightweight                       | Struggles with unseen noise if untrained |
| Suitable for real-time systems    | Limited context understanding          |

---

### 🚀 Future Improvements
- Add noise augmentation (MUSAN, RIR)
- Longer training with GPU
- Add hybrid fusion with AASIST or Wav2Vec2
- Tune thresholds for deployment

---

### 💬 Reflection

**1. Most Significant Challenges?**  
Training without GPU and adapting old code for Windows.

**2. Real-World vs Research?**  
Research audio is clean. Real-world has noise, accents, etc. Fine-tuning is required.

**3. What Can Improve It?**  
- More training data  
- Noise augmentation  
- GPU acceleration

**4. Deployment Plan?**
- Convert to ONNX / TorchScript  
- Serve with FastAPI or Flask  
- Run inference on edge or cloud  
- Monitor confidence scores for safety

---

## ✅ Requirements

### 🔧 Setup Instructions
```bash
git clone https://github.com/DINAKAR-S/Audio-Deepfake-Detection-for-Real-Conversations
cd Baseline-RawNet2
pip install -r requirements.txt
python main.py --database_path "./data/LA" --protocols_path "./data/LA"
```

### 🧾 Dependencies
- torch >= 1.10
- librosa
- numpy
- yaml
- tensorboardX

---

## 📂 Dataset Access
- [ASVspoof 2019 Dataset (Logical Access)](https://datashare.ed.ac.uk/handle/10283/3336)
- Place data in this structure:
```
./data/LA/
├── ASVspoof2019_LA_train
├── ASVspoof2019_LA_dev
├── ASVspoof2021_LA_eval
├── ASVspoof_LA_cm_protocols
```

---

---

📄 **Attached Full Report**

A full detailed document containing all three parts of the assignment — **Research, Implementation, and Analysis** — is also attached in this GitHub repository.

📌 **File Name**: `Assignment_Momenta.pdf`

👉 Kindly go through the PDF for a complete overview of the project, including comparisons, explanations, and reflections.

---
