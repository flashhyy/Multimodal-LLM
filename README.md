# Multimodal-LLM
Prescription Extraction from Handwritten Images using Multimodal LLM (BLIP-2)
# Prescription Information Extraction using Multimodal LLM (BLIP-2)

## About
This project uses a **multimodal large language model (LLM)** — **BLIP-2 + Flan-T5** — to **extract structured information** from **handwritten medical prescription images**.

It automates the extraction of:
- Patient Name
- Doctor Name
- List of Medicines (with Dosage, Frequency, Duration)
- Special Instructions

Outputs are saved as structured JSON files.

---

##  Pipeline Overview
1. Load prescription images from a folder.
2. Preprocess and prepare the images.
3. Use the **BLIP-2 + Flan-T5** model to extract text.
4. Format the results into structured JSON files.
5. Save outputs in an organized way.

---

##  Model Description: Why BLIP-2?
- **BLIP-2** is a **multimodal model** that connects **images** with **language models**.
- It reads **visual content** (messy handwriting, printed text) and **generates text output** directly.
- Combines the power of **vision models** (understanding images) with **language models** (creating structured descriptions).
- **Open-source**, **free**, and **easy to fine-tune** later for specific tasks.

---

##  Evaluation Strategy
Since the dataset does not have ground truth labels:
- **Manual Verification**: Check a random sample of extracted JSON outputs to verify accuracy of:
  - Patient Name
  - Medicines
  - Dosages and Frequencies
  - Special Instructions
- **Optional**: Create a small **gold dataset** with manually annotated labels, and calculate:
  - Precision
  - Recall
  - F1 Score
- **Qualitative Analysis**: Group common errors (e.g., illegible handwriting, missing fields) and analyze.

---

##  Installation

1. Clone this repository:
   ```bash

