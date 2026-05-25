
import os
import time
import pickle
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, roc_curve
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Early Stopping Utility
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, save_path="best_model.pth"):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"Validation loss decreased. Saving model to {self.save_path}")
        else:
            self.counter += 1
            print(f"No improvement. EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# Dataset
class ITM_Dataset(Dataset):
    def __init__(self, images_path, data_file, sentence_embeddings, data_split, train_ratio=1.0):
        self.images_path = images_path
        self.data_file = data_file
        self.sentence_embeddings = sentence_embeddings
        self.data_split = data_split.lower()
        self.train_ratio = train_ratio if self.data_split == "train" else 1.0

        self.image_data = []
        self.question_data = []
        self.answer_data = []
        self.question_embeddings_data = []
        self.answer_embeddings_data = []
        self.label_data = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.load_data()

    def load_data(self):
        random.seed(42)
        with open(self.data_file) as f:
            lines = f.readlines()
            if self.data_split == "train":
                random.shuffle(lines)
                num_samples = int(len(lines) * self.train_ratio)
                lines = lines[:num_samples]
            for line in lines:
                img_name, text, raw_label = line.strip().split("\t")
                img_path = os.path.join(self.images_path, img_name.strip())
                question_text, answer_text = text.split("?")[0].strip() + '?', text.split("?")[1].strip()
                label = 1 if raw_label == "match" else 0
                self.image_data.append(img_path)
                self.question_data.append(question_text)
                self.answer_data.append(answer_text)
                self.question_embeddings_data.append(self.sentence_embeddings[question_text])
                self.answer_embeddings_data.append(self.sentence_embeddings[answer_text])
                self.label_data.append(label)

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img = Image.open(self.image_data[idx]).convert("RGB")
        img = self.transform(img)
        question_embedding = torch.tensor(self.question_embeddings_data[idx], dtype=torch.float32)
        answer_embedding = torch.tensor(self.answer_embeddings_data[idx], dtype=torch.float32)
        label = torch.tensor(self.label_data[idx], dtype=torch.long)
        return img, question_embedding, answer_embedding, label

def load_sentence_embeddings(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

class ITM_Model(nn.Module):
    def __init__(self, num_classes=2, ARCHITECTURE=None, PRETRAINED=None):
        super(ITM_Model, self).__init__()
        if ARCHITECTURE == "CNN":
            self.vision_model = models.resnet18(pretrained=PRETRAINED)
            if PRETRAINED:
                for param in self.vision_model.parameters():
                    param.requires_grad = False
                for param in list(self.vision_model.children())[-2:]:
                    for p in param.parameters():
                        p.requires_grad = True
            self.vision_model.fc = nn.Linear(self.vision_model.fc.in_features, 128)
        else:
            exit(0)
        self.question_embedding_layer = nn.Linear(768, 128)
        self.answer_embedding_layer = nn.Linear(768, 128)
        self.fc = nn.Linear(128 * 3, num_classes)

    def forward(self, img, question_embedding, answer_embedding):
        img_features = self.vision_model(img)
        question_features = self.question_embedding_layer(question_embedding)
        answer_features = self.answer_embedding_layer(answer_embedding)
        combined = torch.cat((img_features, question_features, answer_features), dim=1)
        return self.fc(combined)

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, q_embed, a_embed, labels in val_loader:
            images, q_embed, a_embed, labels = images.to(device), q_embed.to(device), a_embed.to(device), labels.to(device)
            outputs = model(images, q_embed, a_embed)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def train_model(model, ARCHITECTURE, train_loader, val_loader, criterion, optimiser, num_epochs=20):
    print(f'TRAINING {ARCHITECTURE} model')
    model.train()
    early_stopping = EarlyStopping(patience=3, save_path="best_model.pth")
    for epoch in range(num_epochs):
        total_loss = 0.0
        start_time = time.time()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", dynamic_ncols=True, leave=True)
        for images, q_embed, a_embed, labels in loop:
            images, q_embed, a_embed, labels = images.to(device), q_embed.to(device), a_embed.to(device), labels.to(device)
            optimiser.zero_grad()
            outputs = model(images, q_embed, a_embed)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        loop.set_postfix(avg_epoch_loss=avg_loss)
        print(f"Epoch {epoch+1}, Avg Train Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")
        val_loss = validate_model(model, val_loader, criterion)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

def evaluate_model(model, ARCHITECTURE, test_loader, device):
    print(f"EVALUATING {ARCHITECTURE} model")
    model.eval()
    all_labels, all_preds, all_probs, prediction_times = [], [], [], []
    start_time = time.time()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, q_embed, a_embed, labels in tqdm(test_loader, desc="Evaluating", dynamic_ncols=True, leave=True):
            images, q_embed, a_embed, labels = images.to(device), q_embed.to(device), a_embed.to(device), labels.to(device)
            pred_start = time.time()
            outputs = model(images, q_embed, a_embed)
            pred_end = time.time()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            prediction_times.extend([pred_end - pred_start] * labels.size(0))
            preds = outputs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs)

    total_time = time.time() - start_time
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    mrr_total = sum(1.0 / (np.argsort(all_preds)[::-1].tolist().index(l) + 1) if l in all_preds else 0 for l in all_labels)
    mrr = mrr_total / len(all_labels)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Total Test Time: {total_time:.2f}s")
    print(f"Average Prediction Time per Sample: {np.mean(prediction_times):.4f}s")

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig("confusion_matrix_heatmap.png")
    plt.close()

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig("roc_curve_plot.png")
    plt.close()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected Device: {device}")

    IMAGES_PATH = "./visual7w-images"
    train_data_file = "./visual7w-text/v7w.TrainImages.itm.txt"
    test_data_file = "./visual7w-text/v7w.TestImages.itm.txt"
    sentence_embeddings_file = "v7w.sentence_embeddings-gtr-t5-large.pkl"

    sentence_embeddings = load_sentence_embeddings(sentence_embeddings_file)
    full_dataset = ITM_Dataset(IMAGES_PATH, train_data_file, sentence_embeddings, data_split="train", train_ratio=1.0)
    train_len = int(0.8 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_dataset = ITM_Dataset(IMAGES_PATH, test_data_file, sentence_embeddings, data_split="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = ITM_Model(num_classes=2, ARCHITECTURE="CNN", PRETRAINED=True).to(device)
    print("\nModel Architecture:\n", model)
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

    train_model(model, "CNN", train_loader, val_loader, criterion, optimiser, num_epochs=20)
    model.load_state_dict(torch.load("best_model.pth"))  # Load best model
    evaluate_model(model, "CNN", test_loader, device)
