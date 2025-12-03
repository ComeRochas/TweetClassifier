import torch
import torch.nn as nn
from transformers import AutoModel

class MultimodalTweetClassifier(nn.Module):
    def __init__(self, 
                 transformer_name="cardiffnlp/twitter-xlm-roberta-base",
                 metadata_dim=8,
                 text_hidden_dim=768,
                 meta_hidden_dim=32,
                 fusion_hidden_dim=256):
        super(MultimodalTweetClassifier, self).__init__()
        
        # 1. Transformer Encoder
        self.transformer = AutoModel.from_pretrained(transformer_name)
        
        # 2. Metadata MLP
        self.meta_mlp = nn.Sequential(
            nn.Linear(metadata_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(meta_hidden_dim)
        )
        
        # 3. Fusion Classifier
        self.classifier = nn.Sequential(
            nn.Linear(text_hidden_dim + meta_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, 2)  # logits for 2 classes
        )

    def forward(self, input_ids, attention_mask, metadata):
        # Pass text through transformer
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Take CLS embedding (first token)
        # outputs.last_hidden_state shape: (batch_size, seq_len, hidden_dim)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Pass metadata through MLP
        meta_repr = self.meta_mlp(metadata)
        
        # Concatenate
        fused = torch.cat([cls_embedding, meta_repr], dim=1)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits
