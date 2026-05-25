import os
import time
import random
import pickle
import torch
import numpy as np
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix,roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

# Dataset
class ITM_Dataset(Dataset):
    def __init__(self, images_path, data_file, data_split="train", train_ratio=1.0):
        self.images_path = images_path
        self.data_file = data_file
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
        with open(self.data_file) as f:
            lines = f.readlines()
            if self.data_split == "train":
                random.shuffle(lines)
                lines = lines[:int(len(lines) * self.train_ratio)]
            for line in lines:
                img_name, text, raw_label = line.strip().split("\t")
                question, answer = text.split("?")
                question = question.strip() + "?"
                answer = answer.strip()
                label = 1 if raw_label == "match" else 0

                self.image_data.append(os.path.join(self.images_path, img_name))
                self.question_data.append(question)
                self.answer_data.append(answer)
                self.label_data.append(label)

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img = Image.open(self.image_data[idx]).convert("RGB")
        img = self.transform(img)

        question_text = self.question_data[idx]
        answer_text = self.answer_data[idx]

        with torch.no_grad():
            q_tokens = bert_tokenizer(question_text, return_tensors="pt", truncation=True, padding=True).to(device)
            a_tokens = bert_tokenizer(answer_text, return_tensors="pt", truncation=True, padding=True).to(device)
            q_embed = bert_model(**q_tokens).last_hidden_state[:, 0, :].squeeze(0).cpu()
            a_embed = bert_model(**a_tokens).last_hidden_state[:, 0, :].squeeze(0).cpu()

        label = torch.tensor(self.label_data[idx], dtype=torch.long)
        return img, q_embed, a_embed, label

# Model
class ITM_Model(nn.Module):
    def __init__(self, text_dim=768, img_dim=512, hidden_dim=128, num_classes=2):
        super().__init__()
        self.vision_model = models.resnet18(pretrained=True)
        self.vision_model.fc = nn.Identity()

        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.q_proj = nn.Linear(text_dim, hidden_dim)
        self.a_proj = nn.Linear(text_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, q_emb, a_emb):
        img_feat = self.vision_model(img)
        img_proj = self.img_proj(img_feat)
        q_proj = self.q_proj(q_emb)
        a_proj = self.a_proj(a_emb)
        fused = torch.cat([img_proj, q_proj, a_proj], dim=1)
        return self.classifier(fused)

# Training
def train_model(model, train_loader, criterion, optimizer, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for img, q_emb, a_emb, labels in train_loader:
            img, q_emb, a_emb, labels = img.to(device), q_emb.to(device), a_emb.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(img, q_emb, a_emb)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Evaluation
# Evaluation
def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    total_test_time = 0
    total_pred_time = 0
    all_labels, all_preds = [], []

    start_time = time.time()  # Start timer for total test time
    with torch.no_grad():
        for img, q_emb, a_emb, labels in tqdm(test_loader, desc="Evaluating", ncols=100, dynamic_ncols=True):
            img, q_emb, a_emb, labels = img.to(device), q_emb.to(device), a_emb.to(device), labels.to(device)

            # Track prediction time per sample
            pred_start_time = time.time()
            outputs = model(img, q_emb, a_emb)
            pred_end_time = time.time()
            total_pred_time += (pred_end_time - pred_start_time)

            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    total_test_time = time.time() - start_time  # Total test time

    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    avg_pred_time_per_sample = total_pred_time / len(test_loader.dataset)

    # Print metrics
    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Average Evaluation Loss: {avg_loss:.4f}")
    print(f"Total Test Time: {total_test_time:.2f}s")
    print(f"Average Prediction Time per Sample: {avg_pred_time_per_sample:.6f}s")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Save Confusion Matrix Heatmap
    if not os.path.exists('./output_plots'):
        os.makedirs('./output_plots')
    conf_matrix_file = './output_plots/confusion_matrix.png'
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Match', 'Match'], yticklabels=['No Match', 'Match'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(conf_matrix_file)
    plt.close()

    # Save ROC-AUC curve
    roc_auc_file = './output_plots/roc_auc_curve.png'
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

    # Compute MRR (Mean Reciprocal Rank)
    mrr = np.mean([1 / (i + 1) if all_preds[i] == all_labels[i] else 0 for i in range(len(all_labels))])
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")


# Main
if __name__ == '__main__':
    IMAGES_PATH = "./visual7w-images"
    TRAIN_FILE = "./visual7w-text/v7w.TrainImages.itm.txt"
    TEST_FILE = "./visual7w-text/v7w.TestImages.itm.txt"

    train_dataset = ITM_Dataset(IMAGES_PATH, TRAIN_FILE, data_split="train", train_ratio=0.2)
    test_dataset = ITM_Dataset(IMAGES_PATH, TEST_FILE, data_split="test")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = ITM_Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

    train_model(model, train_loader, criterion, optimizer, num_epochs=1)
    evaluate_model(model, test_loader, criterion)
