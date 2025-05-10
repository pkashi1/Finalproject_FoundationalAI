import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import os

# Set random seeds for reproducibility across different runs
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

df = pd.read_csv("analysis_copy_sorted.csv")
target_cols = ['area', 'rg', 'rdf']
df.dropna(subset=['Input_List'] + target_cols, inplace=True)
for col in target_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=target_cols, inplace=True)
def extract_tokens(input_str):
    return [token.strip() for _, token in ast.literal_eval(input_str)]

# Process sequences and create vocabulary mapping
df['sequence'] = df['Input_List'].apply(extract_tokens)

all_tokens = set(token for seq in df['sequence'] for token in seq)
# print(df.head())
# print(f"Unique tokens: {len(all_tokens)}")
# print(all_tokens) 
token2idx = {token: idx + 2 for idx, token in enumerate(all_tokens)}
token2idx['PAD'] = 0  
token2idx['CLS'] = 1  
vocab_size = len(token2idx)
max_len = 22  

def encode_sequence(seq):
    tokens = [token2idx['CLS']] + [token2idx[token] for token in seq]
    return tokens[:max_len] + [token2idx['PAD']] * (max_len - len(tokens))

df['encoded'] = df['sequence'].apply(encode_sequence)
# print(df['encoded'][:5]) 
# Custom Dataset class for handling polymer sequences and their properties
class PolymerDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        print(f"Sequence tensor shape: {self.sequences.shape}")
        self.targets = torch.tensor(targets, dtype=torch.float32)
        print(f"Target tensor shape: {self.targets.shape}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# PositionalEncoding class adds positional information to input embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
#Main Transformer model for regression tasks
class TransformerRegressor(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=1, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.head(x)
# Function to calculate adjusted R² score
def adjusted_r2_score(y_true, y_pred, n, p):
    r2 = r2_score(y_true, y_pred)
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
# Function to train and evaluate the model for each target property
def train_and_evaluate(target_name):
    print(f"\nTraining model for target: {target_name}")
    scaler = RobustScaler()
    y_scaled = scaler.fit_transform(df[[target_name]].values.astype(np.float32))
    dataset = PolymerDataset(df['encoded'].tolist(), y_scaled)
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.4)
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    device = torch.device('cpu')
    model = TransformerRegressor(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    
    for epoch in range(25):
        model.train()
        total_loss = 0
        for seqs, targets in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seqs, targets in val_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                outputs = model(seqs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"best_model_{target_name}.pth")
        print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Loss Curve - {target_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"loss_{target_name}.png")
    plt.clf()
    model.load_state_dict(torch.load(f"best_model_{target_name}.pth"))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for seqs, targets in val_loader:
            seqs = seqs.to(device)
            preds = model(seqs).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(targets.numpy())

    y_true = scaler.inverse_transform(np.array(y_true))
    y_pred = scaler.inverse_transform(np.array(y_pred))

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    n = len(y_true) 
    p = 1 
    adj_r2 = adjusted_r2_score(y_true, y_pred, n, p)

    print(f"\nFinal Evaluation for {target_name}: MSE = {mse:.2f}, R² = {r2:.3f}, Adjusted R² = {adj_r2:.3f}")
for target in target_cols:
    train_and_evaluate(target)