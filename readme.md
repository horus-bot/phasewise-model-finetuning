# Machine Learning Model Training & Fine-Tuning Experiment

This repository contains an end-to-end machine learning training pipeline that explores **model training, validation, and fine-tuning (Phase 1 & Phase 2)**.  
The project focuses on understanding **generalization, overfitting, and validation behavior** rather than achieving state-of-the-art performance.

---

## 📌 Overview

The project implements:
- A complete training loop with accuracy tracking
- A validation pipeline to monitor generalization
- A two-phase training strategy:
  - **Phase 1:** Base training
  - **Phase 2:** Fine-tuning with additional parameters unfrozen

The goal is to study how fine-tuning impacts training vs validation performance.

---

## 🧠 Training Strategy

### Phase 1
- Initial training phase
- Model learns core feature representations
- Validation accuracy improves steadily in early epochs

### Phase 2 (Fine-Tuning)
- Additional layers are unfrozen
- Model capacity increases
- Training accuracy improves further, while validation accuracy fluctuates

This setup highlights a common real-world challenge: **improved training performance does not always translate to better generalization**.

---

## 📊 Results Summary

| Metric | Observation |
|------|------------|
| Training Accuracy | Increased steadily from ~43% to ~86% |
| Validation Accuracy | Peaked early (~61%) and then fluctuated |
| Best Epoch | Early training (around Epoch 3–4) |
| Final Outcome | Signs of overfitting during fine-tuning |

**Key Insight:**  
The best-performing model is **not the final epoch**, but an earlier checkpoint where validation accuracy was highest. This demonstrates the importance of **early stopping and checkpoint selection**.

---

## 🔍 Interpretation

- The model successfully fits the training data
- Validation performance indicates limited generalization
- Fine-tuning improves memorization more than real-world performance
- The project serves as a **learning-focused experiment**, not a production-ready system

---

## ✅ What This Project Demonstrates Well

- Proper separation of training and validation data
- Transparent reporting of results (including weaknesses)
- Realistic behavior of models under fine-tuning
- Practical ML experimentation workflow

---

## ⚠️ Limitations

- Model is **not production-ready**
- Validation accuracy is unstable
- Performance depends heavily on early checkpoint selection
- Further improvements would require more data, regularization, or architectural tuning

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
