import torch
from torch.utils.data import Dataset

class TweetDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, with_labels=True):
        """
        Args:
            data (list of dict): List of samples. Each sample is a dict with:
                - "text": str
                - "metadata": np.ndarray or list of shape (8,)
                - "label": int (optional, if with_labels=True)
            tokenizer: HuggingFace tokenizer
            max_length (int): Maximum sequence length
            with_labels (bool): Whether the data contains labels
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        metadata = item["metadata"]

        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Squeeze the batch dimension (1, seq_len) -> (seq_len)
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Convert metadata to float tensor
        metadata_tensor = torch.tensor(metadata, dtype=torch.float32)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "metadata": metadata_tensor
        }

        if self.with_labels:
            label = item["label"]
            result["labels"] = torch.tensor(label, dtype=torch.long)

        return result
