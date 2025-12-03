import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle
import random
import os

import argparse
import pprint
from dataset import TweetDataset
from model import MultimodalTweetClassifier
from config import get_default_config

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_features_from_row(row):
    # 1. Extract Full Text
    text = row.get('text', '')
    # Handle nested extended_tweet.full_text if it exists in the flattened row
    # Note: When using json_normalize, keys become 'extended_tweet.full_text'
    if pd.notna(row.get('extended_tweet.full_text')):
        text = row['extended_tweet.full_text']
        
    # normalize text and append user URL tag
    text = str(text).replace('\n', ' ').strip()
    user_url = row.get('user.url', '')
    if pd.notna(user_url) and user_url:
        text = f"[TEXTE] {text} [PERSONAL_URL] {user_url}"
    else:
        text = f"[TEXTE] {text}"
        
    # 2. Extract Metadata
    def safe_get(key, default=0):
        val = row.get(key, default)
        if pd.isna(val):
            return default
        return val

    metadata = {
        'user_default_profile': int(safe_get('user.default_profile', False)),
        'user_profile_use_background_image': int(safe_get('user.profile_use_background_image', False)),
        'user_statuses_count': float(safe_get('user.statuses_count', 0)),
        'user_profile_background_tile': int(safe_get('user.profile_background_tile', False)),
        'user_geo_enabled': int(safe_get('user.geo_enabled', False)),
        'user_is_translator': int(safe_get('user.is_translator', False)),
        'user_favourites_count': float(safe_get('user.favourites_count', 0)),
        'user_listed_count': float(safe_get('user.listed_count', 0))
    }
    
    return pd.Series({'full_text': text, **metadata})

def load_and_preprocess_train_data(file_path):
    print(f"Loading data from {file_path}...")
    # Read JSONL
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Normalize
    df = pd.json_normalize(data)
    
    # Extract features
    print("Extracting features...")
    features = df.apply(extract_features_from_row, axis=1)
    
    # Log transform
    log_cols = ['user_statuses_count', 'user_favourites_count', 'user_listed_count']
    for col in log_cols:
        features[col] = np.log1p(features[col])
        
    # Metadata columns
    metadata_cols = [c for c in features.columns if c != 'full_text']
    metadata = features[metadata_cols].values
    
    # Scale
    print("Scaling metadata...")
    scaler = StandardScaler()
    metadata_scaled = scaler.fit_transform(metadata)
    
    # Save scaler for inference
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved to scaler.pkl")
    
    # Prepare final list of dicts
    labels = df['label'].values
    texts = features['full_text'].values
    
    training_data = []
    for i in range(len(df)):
        training_data.append({
            "text": texts[i],
            "metadata": metadata_scaled[i],
            "label": labels[i]
        })
        
    return training_data

def train():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Multimodal Tweet Classifier")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr_transformer", type=float, help="Learning rate for transformer")
    parser.add_argument("--lr_head", type=float, help="Learning rate for head")
    parser.add_argument("--max_length", type=int, help="Max sequence length")
    parser.add_argument("--freeze_transformer", action='store_true', help="Freeze transformer weights (no fine-tuning)")
    args = parser.parse_args()

    # Build config
    cfg = get_default_config()
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.num_epochs is not None: cfg.num_epochs = args.num_epochs
    if args.lr_transformer is not None: cfg.lr_transformer = args.lr_transformer
    if args.lr_head is not None: cfg.lr_head = args.lr_head
    if args.max_length is not None: cfg.max_length = args.max_length
    if args.freeze_transformer: cfg.freeze_transformer = True

    print("Training config:")
    pprint.pprint(cfg)

    set_seed(cfg.seed)
    
    # Load Data
    if os.path.exists('train.jsonl'):
        training_data = load_and_preprocess_train_data('train.jsonl')
    else:
        print("Error: train.jsonl not found.")
        return

    # Split Train/Val
    train_list, val_list = train_test_split(training_data, test_size=0.2, random_state=cfg.seed, stratify=[x['label'] for x in training_data])
    
    print(f"Train size: {len(train_list)}, Val size: {len(val_list)}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.transformer_name)
    
    # Datasets
    train_dataset = TweetDataset(train_list, tokenizer, cfg.max_length, with_labels=True)
    val_dataset = TweetDataset(val_list, tokenizer, cfg.max_length, with_labels=True)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MultimodalTweetClassifier(transformer_name=cfg.transformer_name)
    model.to(device)
    
    # Optionally freeze transformer parameters
    if cfg.freeze_transformer:
        print("Freezing transformer parameters...")
        for p in model.transformer.parameters():
            p.requires_grad = False
    
    # Optimizer
    # Build optimizer parameter groups only for parameters that require_grad
    transformer_params = [p for p in model.transformer.parameters() if p.requires_grad]
    param_groups = []
    if len(transformer_params) > 0:
        param_groups.append({'params': transformer_params, 'lr': cfg.lr_transformer})
    param_groups.append({'params': model.meta_mlp.parameters(), 'lr': cfg.lr_head})
    param_groups.append({'params': model.classifier.parameters(), 'lr': cfg.lr_head})
    optimizer = AdamW(param_groups, weight_decay=cfg.weight_decay)

    # Debug: print parameter counts and optimizer groups
    def param_counts(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        transf_total = sum(p.numel() for p in model.transformer.parameters())
        transf_trainable = sum(p.numel() for p in model.transformer.parameters() if p.requires_grad)
        meta_total = sum(p.numel() for p in model.meta_mlp.parameters())
        classifier_total = sum(p.numel() for p in model.classifier.parameters())
        print(f"Total params: {total:,}, trainable: {trainable:,}")
        print(f"Transformer params: {transf_total:,}, trainable: {transf_trainable:,}")
        print(f"Meta MLP params: {meta_total:,}, Classifier params: {classifier_total:,}")
    param_counts(model)
    for i, g in enumerate(optimizer.param_groups):
        n = sum(p.numel() for p in g['params'])
        lr = g.get('lr', None)
        print(f"Optimizer group {i}: params={n:,}, lr={lr}, weight_decay={g.get('weight_decay')}")
    
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    best_val_acc = 0.0
    
    for epoch in range(cfg.num_epochs):
        print(f"\nEpoch {epoch+1}/{cfg.num_epochs}")
        
        # Train
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, metadata)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                metadata = batch['metadata'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids, attention_mask, metadata)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_multimodal_model.pt")
            print("Saved new best model.")

if __name__ == "__main__":
    train()
