import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple

class NL2BashDataset(Dataset):
    """Dataset class for NL2Bash data with proper tokenization and padding."""
    
    def __init__(self, data_path: str, tokenizer_name: str = "t5-small", max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Add special tokens if they don't exist
        special_tokens = ["<CMD>", "</CMD>", "<NL>", "</NL>"]
        self.tokenizer.add_tokens(special_tokens)
        
        # Load and preprocess data
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load and preprocess the JSON data."""
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        processed_data = []
        for key, value in raw_data.items():
            # Clean the text
            invocation = value['invocation'].strip()
            cmd = value['cmd'].strip()
            
            # Add special tokens for better sequence-to-sequence learning
            input_text = f"<NL> {invocation} </NL>"
            target_text = f"<CMD> {cmd} </CMD>"
            
            processed_data.append({
                'input_text': input_text,
                'target_text': target_text,
                'invocation': invocation,
                'cmd': cmd
            })
        
        return processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize input and target
        input_encoding = self.tokenizer(
            item['input_text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            item['target_text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
            'target_attention_mask': target_encoding['attention_mask'].squeeze(),
            'invocation': item['invocation'],
            'cmd': item['cmd']
        }

def create_data_loaders(data_path: str, tokenizer_name: str = "t5-small", 
                       batch_size: int = 16, test_size: float = 0.2, 
                       random_state: int = 42) -> Tuple[DataLoader, DataLoader, AutoTokenizer]:
    """Create train and validation data loaders."""
    
    # Load dataset
    dataset = NL2BashDataset(data_path, tokenizer_name)
    
    # Split into train and validation
    train_indices, val_indices = train_test_split(
        list(range(len(dataset))), 
        test_size=test_size, 
        random_state=random_state
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, dataset.tokenizer

def get_data_statistics(data_path: str):
    """Get basic statistics about the dataset."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    invocations = [v['invocation'] for v in data.values()]
    commands = [v['cmd'] for v in data.values()]
    
    stats = {
        'total_samples': len(data),
        'avg_invocation_length': sum(len(inv.split()) for inv in invocations) / len(invocations),
        'avg_command_length': sum(len(cmd.split()) for cmd in commands) / len(commands),
        'max_invocation_length': max(len(inv.split()) for inv in invocations),
        'max_command_length': max(len(cmd.split()) for cmd in commands),
    }
    
    return stats

if __name__ == "__main__":
    # Test the data preprocessing
    stats = get_data_statistics("nl2bash-data.json")
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test data loader
    train_loader, val_loader, tokenizer = create_data_loaders("nl2bash-data.json", batch_size=4)
    
    print(f"\nTrain samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Show a sample
    sample_batch = next(iter(train_loader))
    print(f"\nSample batch shapes:")
    print(f"  Input IDs: {sample_batch['input_ids'].shape}")
    print(f"  Labels: {sample_batch['labels'].shape}")
    print(f"\nSample invocation: {sample_batch['invocation'][0]}")
    print(f"Sample command: {sample_batch['cmd'][0]}") 