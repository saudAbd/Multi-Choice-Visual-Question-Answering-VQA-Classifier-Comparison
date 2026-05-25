import os
import time
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_curve

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Selected Device: {device}")

# Dataset
class ITM_Dataset(Dataset):
    def __init__(self, images_path, data_file, tokenizer, data_split, train_ratio=1.0):
        self.images_path = images_path
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.data_split = data_split.lower()
        self.train_ratio = train_ratio if self.data_split == "train" else 1.0

        self.image_data = []
        self.question_data = []
        self.answer_data = []
        self.label_data = []

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.load_data()

    def load_data(self):
        with open(self.data_file, 'r') as f:
            lines = f.readlines()

        if self.data_split == "train":
            random.shuffle(lines)
            lines = lines[:int(len(lines) * self.train_ratio)]

        for line in lines:
            img_name, text, raw_label = line.strip().split("\t")
            question_text, answer_text = text.split("?", 1)
            question_text = question_text.strip() + "?"
            answer_text = answer_text.strip()
            label = 1 if raw_label == "match" else 0

            img_path = os.path.join(self.images_path, img_name.strip())

            self.image_data.append(img_path)
            self.question_data.append(question_text)
            self.answer_data.append(answer_text)
            self.label_data.append(label)

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img = Image.open(self.image_data[idx]).convert("RGB")
        img = self.transform(img)
        question = self.question_data[idx]
        answer = self.answer_data[idx]
        label = torch.tensor(self.label_data[idx], dtype=torch.long)

        q_tokens = tokenizer(question, return_tensors="pt", padding='max_length', truncation=True, max_length=32)
        a_tokens = tokenizer(answer, return_tensors="pt", padding='max_length', truncation=True, max_length=32)

        return img, q_tokens['input_ids'].squeeze(0), a_tokens['input_ids'].squeeze(0), label


# Text Encoder (LSTM)
class TextLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128):
        super(TextLSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


# Image Encoder (Simple CNN)
class SimpleCNN(nn.Module):
    def __init__(self, output_dim=128):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 56 * 56, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Full Model
class ITM_Model(nn.Module):
    def __init__(self, vocab_size):
        super(ITM_Model, self).__init__()
        self.image_encoder = SimpleCNN(128)
        self.text_encoder = TextLSTMEncoder(vocab_size)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, img, q_ids, a_ids):
        img_feat = self.image_encoder(img)
        q_feat = self.text_encoder(q_ids)
        a_feat = self.text_encoder(a_ids)
        combined = torch.cat([img_feat, q_feat, a_feat], dim=1)
        return self.classifier(combined)


# Training

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        start_time = time.time()  # Time for the epoch
        total_loss = 0.0
        for img, q_ids, a_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100, dynamic_ncols=True):
            img, q_ids, a_ids, labels = img.to(device), q_ids.to(device), a_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(img, q_ids, a_ids)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Time: {epoch_time:.2f}s")


# Evaluation

def evaluate_model(model, test_loader, save_dir='./output_plots'):
    model.eval()
    all_labels, all_preds = [], []
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_test_time = 0.0
    total_pred_time = 0.0

    start_time = time.time()
    with torch.no_grad():
        for img, q_ids, a_ids, labels in tqdm(test_loader, desc="Evaluating", ncols=100, dynamic_ncols=True):
            img, q_ids, a_ids, labels = img.to(device), q_ids.to(device), a_ids.to(device), labels.to(device)
            pred_start_time = time.time()
            output = model(img, q_ids, a_ids)
            pred_end_time = time.time()
            total_pred_time += (pred_end_time - pred_start_time)

            loss = criterion(output, labels)
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    total_test_time = time.time() - start_time
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    avg_loss = total_loss / len(test_loader)
    avg_pred_time_per_sample = total_pred_time / len(test_loader.dataset)
    mrr = np.mean([1 / (i + 1) if all_preds[i] == all_labels[i] else 0 for i in range(len(all_labels))])

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Average Evaluation Loss: {avg_loss:.4f}")
    print(f"Total Test Time: {total_test_time:.2f}s")
    print(f"Average Prediction Time per Sample: {avg_pred_time_per_sample:.6f}s")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Save Confusion Matrix Heatmap
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    conf_matrix_file = os.path.join(save_dir, 'confusion_matrix.png')
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Match', 'Match'], yticklabels=['No Match', 'Match'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(conf_matrix_file)
    plt.close()

    # Save ROC-AUC curve
    roc_auc_file = os.path.join(save_dir, 'roc_auc_curve.png')
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(roc_auc_file)
    plt.close()


# Main
if __name__ == '__main__':
    IMAGES_PATH = "./visual7w-images"
    train_file = "./visual7w-text/v7w.TrainImages.itm.txt"
    test_file = "./visual7w-text/v7w.TestImages.itm.txt"

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    train_dataset = ITM_Dataset(IMAGES_PATH, train_file, tokenizer, "train", train_ratio=0.2)
    test_dataset = ITM_Dataset(IMAGES_PATH, test_file, tokenizer, "test")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = ITM_Model(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    train_model(model, train_loader, criterion, optimizer, epochs=1)
    evaluate_model(model, test_loader)
