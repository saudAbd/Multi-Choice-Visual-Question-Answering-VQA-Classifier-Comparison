import os
import time
import pickle
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vit_b_32
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Custom Dataset
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
        print("LOADING data from "+str(self.data_file))
        print("=========================================")
        random.seed(42)
        with open(self.data_file) as f:
            lines = f.readlines()
            if self.data_split == "train":
                random.shuffle(lines)
                num_samples = int(len(lines) * self.train_ratio)
                lines = lines[:num_samples]
            for line in lines:
                line = line.rstrip("\n")
                img_name, text, raw_label = line.split("\t")
                img_path = os.path.join(self.images_path, img_name.strip())
                question_answer_text = text.split("?")
                question_text = question_answer_text[0].strip() + '?' 
                answer_text = question_answer_text[1].strip()
                label = 1 if raw_label == "match" else 0
                self.image_data.append(img_path)
                self.question_data.append(question_text)
                self.answer_data.append(answer_text)
                self.question_embeddings_data.append(self.sentence_embeddings[question_text])
                self.answer_embeddings_data.append(self.sentence_embeddings[answer_text])
                self.label_data.append(label)
        print("|image_data|="+str(len(self.image_data)))
        print("|question_data|="+str(len(self.question_data)))
        print("|answer_data|="+str(len(self.answer_data)))
        print("done loading data...")

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img_path = self.image_data[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        question_embedding = torch.tensor(self.question_embeddings_data[idx], dtype=torch.float32)
        answer_embedding = torch.tensor(self.answer_embeddings_data[idx], dtype=torch.float32)
        label = torch.tensor(self.label_data[idx], dtype=torch.long)
        return img, question_embedding, answer_embedding, label

# Load sentence embeddings
def load_sentence_embeddings(file_path):
    print("READING sentence embeddings...")
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# ViT Encoder
class Transformer_VisionEncoder(nn.Module):
    def __init__(self, pretrained=None):
        super(Transformer_VisionEncoder, self).__init__()
        if pretrained:
            self.vision_model = vit_b_32(weights="IMAGENET1K_V1")
            for param in self.vision_model.parameters():
                param.requires_grad = False
            for param in list(self.vision_model.heads.parameters())[-2:]:
                param.requires_grad = True
        else:
            self.vision_model = vit_b_32(weights=None)
        self.num_features = self.vision_model.heads[0].in_features
        self.vision_model.heads = nn.Identity()

    def forward(self, x):
        return self.vision_model(x)

# Model
class ITM_Model(nn.Module):
    def __init__(self, num_classes=2, ARCHITECTURE=None, PRETRAINED=None):
        super(ITM_Model, self).__init__()
        self.ARCHITECTURE = ARCHITECTURE
        if ARCHITECTURE == "ViT":
            self.vision_model = Transformer_VisionEncoder(pretrained=PRETRAINED)
            self.fc_vit = nn.Linear(self.vision_model.num_features, 128)
        else:
            exit(0)
        self.question_embedding_layer = nn.Linear(768, 128)
        self.answer_embedding_layer = nn.Linear(768, 128)
        self.fc = nn.Linear(128 * 3, num_classes)

    def forward(self, img, question_embedding, answer_embedding):
        img_features = self.vision_model(img)
        if self.ARCHITECTURE == "ViT":
            img_features = self.fc_vit(img_features)
        question_features = self.question_embedding_layer(question_embedding)
        answer_features = self.answer_embedding_layer(answer_embedding)
        combined = torch.cat((img_features, question_features, answer_features), dim=1)
        return self.fc(combined)

# Training
def train_model(model, ARCHITECTURE, train_loader, criterion, optimiser, num_epochs=20):
    print(f'TRAINING {ARCHITECTURE} model')
    model.train()
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
        print(f"Epoch {epoch+1}, Avg Train Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")

# Evaluation
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

            prediction_times.append(pred_end - pred_start)
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

    # Mean Reciprocal Rank (MRR)
    mrr_total = 0.0
    for i, pred in enumerate(all_preds):
        sorted_preds = np.argsort(all_preds)[::-1]
        if all_labels[i] in sorted_preds:
            rank = np.where(sorted_preds == all_labels[i])[0][0] + 1
            mrr_total += 1.0 / rank
    mrr = mrr_total / len(all_labels)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Evaluation Loss: {avg_loss:.4f}")
    print(f"Total Test Time: {total_time:.2f}s")
    print(f"Average Prediction Time per Sample: {np.mean(prediction_times):.4f}s")

    # Save Confusion Matrix
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(plot_dir, "confusion_matrix_heatmap.png"))
    plt.close()  # Close after saving the plot

    # Save ROC Curve Plot
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(plot_dir, "roc_curve_plot.png"))
    plt.close()  # Close the plot

# Main
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected Device: {device}")
    
    IMAGES_PATH = "./visual7w-images"
    train_data_file = "./visual7w-text/v7w.TrainImages.itm.txt"
    test_data_file = "./visual7w-text/v7w.TestImages.itm.txt"
    sentence_embeddings_file = "v7w.sentence_embeddings-gtr-t5-large.pkl"

    sentence_embeddings = load_sentence_embeddings(sentence_embeddings_file)
    train_dataset = ITM_Dataset(IMAGES_PATH, train_data_file, sentence_embeddings, data_split="train", train_ratio=0.2)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataset = ITM_Dataset(IMAGES_PATH, test_data_file, sentence_embeddings, data_split="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    MODEL_ARCHITECTURE = "ViT"
    USE_PRETRAINED_MODEL = True
    model = ITM_Model(num_classes=2, ARCHITECTURE=MODEL_ARCHITECTURE, PRETRAINED=USE_PRETRAINED_MODEL).to(device)
    print("\nModel Architecture:\n", model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {total_params}")
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

    train_model(model, MODEL_ARCHITECTURE, train_loader, criterion, optimiser, num_epochs=20)
    evaluate_model(model, MODEL_ARCHITECTURE, test_loader, device)
