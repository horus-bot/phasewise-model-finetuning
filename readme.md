# Diabetic Retinopathy Fine-Tuning Study

This repository contains an end-to-end machine learning training pipeline that explores **model training, validation, and phase-wise fine-tuning (Phase 1 & Phase 2)** using a medical imaging dataset.  
The project focuses on understanding **generalization, overfitting, and validation behavior**, rather than achieving state-of-the-art performance.

---

## 📌 Overview

This project implements:
- A complete training loop with accuracy tracking
- A validation pipeline to monitor generalization
- A two-phase training strategy:
  - **Phase 1:** Base training
  - **Phase 2:** Fine-tuning with additional parameters unfrozen

The primary goal is to study how fine-tuning affects **training vs validation performance**, especially in limited and imbalanced medical datasets.

---

## 🧠 Training Strategy

### Phase 1
- Initial training phase
- Model learns core feature representations
- Validation accuracy improves steadily during early epochs

### Phase 2 (Fine-Tuning)
- Additional layers are unfrozen
- Model capacity increases
- Training accuracy improves further, while validation accuracy fluctuates

This setup highlights a common real-world challenge:  
**better training performance does not always translate to better generalization**.

---

## 📂 Dataset

This project uses the **IDRiD (Indian Diabetic Retinopathy Image Dataset)**, a publicly available medical imaging dataset focused on diabetic retinopathy analysis.

- **Dataset source:**  
  https://www.kaggle.com/datasets/mariaherrerot/idrid-dataset
- **Data type:** Retinal fundus images
- **Domain:** Medical imaging (diabetic retinopathy)

The dataset is relatively small and class-imbalanced, making generalization challenging and providing a realistic setting to study overfitting and fine-tuning behavior.

---

## 📊 Results Summary

| Metric | Observation |
|------|------------|
| Training Accuracy | Increased steadily from ~43% to ~86% |
| Validation Accuracy | Peaked early (~61%) and then fluctuated |
| Best Epoch | Early training (around Epoch 3–4) |
| Final Outcome | Signs of overfitting during fine-tuning |

**Key Insight:**  
The best-performing model is **not the final epoch**, but an earlier checkpoint where validation accuracy was highest. This demonstrates the importance of **early stopping and validation-based checkpoint selection**.

---

## 🔍 Interpretation

- The model successfully fits the training data
- Validation performance indicates limited generalization
- Fine-tuning improves memorization more than real-world performance
- The project is intended as a **learning-focused experiment**, not a production-ready system

---

## ✅ What This Project Demonstrates Well

- Proper separation of training and validation data
- Transparent reporting of results, including limitations
- Realistic fine-tuning behavior on medical datasets
- A practical and reproducible ML experimentation workflow

---

## ⚠️ Limitations

- Model is **not production-ready**
- Validation accuracy is unstable
- Performance depends heavily on early checkpoint selection
- Improvements would require more data, stronger regularization, or architectural changes

---

## ⚠️ Medical Disclaimer

**This project is for research and educational purposes only.**  
It is **not intended for clinical diagnosis, treatment, or medical decision-making**.

---

## 👤 Author

**Harsh Srivastava**  
GitHub: [horus-bot](https://github.com/horus-bot)  
Contact: **blueharsh2@gmail.com**

---

## 📎 Notes

This repository is intended for:
- Learning and experimentation
- Demonstrating ML training practices
- Showcasing honest evaluation and analysis

Feedback and suggestions are welcome.
