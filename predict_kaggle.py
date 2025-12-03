import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import pickle
import os

from dataset import TweetDataset
from model import MultimodalTweetClassifier
# Import the feature extraction logic to ensure consistency
from train_multimodal import extract_features_from_row

def load_and_preprocess_kaggle_data(file_path, scaler_path='scaler.pkl'):
    print(f"Loading data from {file_path}...")
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
    
    # Scale using the saved scaler
    print(f"Loading scaler from {scaler_path}...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    print("Scaling metadata...")
    metadata_scaled = scaler.transform(metadata)
    
    # Prepare final list
    texts = features['full_text'].values
    challenge_ids = df['challenge_id'].values
    
    kaggle_data = []
    for i in range(len(df)):
        kaggle_data.append({
            "text": texts[i],
            "metadata": metadata_scaled[i],
            "challenge_id": challenge_ids[i]
        })
        
    return kaggle_data

def predict():
    # Hyperparameters
    MAX_LENGTH = 160
    BATCH_SIZE = 32
    TRANSFORMER_NAME = "cardiffnlp/twitter-xlm-roberta-base"
    MODEL_PATH = "best_multimodal_model.pt"
    
    # Load Data
    if os.path.exists('kaggle_test.jsonl'):
        kaggle_data = load_and_preprocess_kaggle_data('kaggle_test.jsonl')
    else:
        print("Error: kaggle_test.jsonl not found.")
        return
        
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
    
    # Dataset
    # Note: with_labels=False
    dataset = TweetDataset(kaggle_data, tokenizer, MAX_LENGTH, with_labels=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MultimodalTweetClassifier(transformer_name=TRANSFORMER_NAME)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_ids = []
    
    print("Starting prediction...")
    with torch.no_grad():
        # We need to track which sample corresponds to which ID.
        # The DataLoader preserves order because shuffle=False.
        # We can iterate through the data list index or just zip with the loader if batch size aligns, 
        # but simpler is to just collect preds and then map to IDs since order is preserved.
        
        batch_start_idx = 0
        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            metadata = batch['metadata'].to(device)
            
            logits = model(input_ids, attention_mask, metadata)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            
            # Get corresponding IDs
            batch_size = input_ids.size(0)
            batch_ids = [item['challenge_id'] for item in kaggle_data[batch_start_idx : batch_start_idx + batch_size]]
            all_ids.extend(batch_ids)
            
            batch_start_idx += batch_size
            
    # Create DataFrame
    results_df = pd.DataFrame({
        "ID": all_ids,
        "Prediction": all_preds
    })
    
    # Save CSV
    output_file = "multimodal_transformer.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    predict()
