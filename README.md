# Multi-Choice Visual Question Answering (VQA) – Classifier Comparison

**Dataset used:**  
The dataset consists of **images with associated questions and multiple-choice answers**. Images were resized and normalized, while questions and answers were tokenized and embedded using pretrained models such as BERT. These embeddings align text features with image features for the VQA task.

---

## 📖 Project Overview
This project addresses the **Multi-Choice Visual Question Answering (VQA) task**, where a model must select the correct answer to a question based on an input image.  
We compared different **machine learning classifiers** and fusion strategies to evaluate performance in terms of **accuracy, F1-score, Mean Reciprocal Rank (MRR), precision, recall, and training/testing time**.

---

## ⚙️ Methodology
### Preprocessing
- Images resized and normalized for CNN/ViT models.  
- Text data tokenized and converted into dense embeddings (BERT and precomputed vectors).  
- Features from both modalities combined for final classification.  

### Models Compared
| Model | Image Encoder | Text Encoder | Fusion Strategy |
|-------|---------------|--------------|----------------|
| **1** | CNN (ResNet18, pretrained) | Precomputed embeddings | MLP Fusion |
| **2** | Vision Transformer (ViT) | Precomputed embeddings | MLP Fusion |
| **3** | CNN (ResNet18) | LSTM | Concatenation + MLP |
| **4** | CNN (ResNet18) | BERT | Concatenation + MLP |
| **5** | Vision Transformer (ViT) | BERT | Concatenation + MLP |

- **ResNet18**: Efficient CNN capturing hierarchical visual features.  
- **ViT**: Captures global context using attention.  
- **BERT/LSTM**: Text encoders for questions and answers.  
- **Fusion (MLP / Concatenation)**: Combines image and text representations.  

---

## 📊 Results
- **Best Model:** **ResNet18 + MLP Fusion** (Model 1)  
  - Accuracy: **0.8094**  
  - Precision: **0.8027**  
  - Recall: **0.8094**  
  - F1-Score: **0.7808**  
  - ROC-AUC: **0.8108**  
  - Efficient and computationally lightweight compared to transformer-based models.  

- **ViT + MLP Fusion (Model 2):** Comparable accuracy but slower due to higher computational cost.  
- **CNN + LSTM (Model 3):** Lowest performance due to sequential text encoding limitations.  
- **CNN + BERT (Model 4):** Moderate performance, but high computational overhead.  
- **ViT + BERT (Model 5):** High cost with little gain over simpler models.  

**Key Insight:**  
Simpler models such as **CNN + MLP Fusion** can outperform complex architectures in VQA when considering the trade-off between **performance and efficiency**.

---

## 📌 References
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. CVPR. [ResNet18]  
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). *An image is worth 16x16 words: Transformers for image recognition at scale*. ICLR. [ViT]  
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of deep bidirectional transformers for language understanding*. NAACL.  
- Tolstikhin, I., Houlsby, N., Kolesnikov, A., et al. (2021). *MLP-Mixer: An all-MLP architecture for vision*. NeurIPS.  
