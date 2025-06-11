import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer
from typing import Optional, Tuple
import math

class MultiHeadAttention(nn.Module):
    """Custom Multi-Head Attention implementation for better understanding."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        assert d_model % n_heads == 0
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Apply linear transformations and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output transformation
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(attended)
        
        return output, attention_weights

class TransformerBlock(nn.Module):
    """Single Transformer block with self-attention and feed-forward layers."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights

class NL2BashTransformer(nn.Module):
    """Custom Transformer model for Natural Language to Bash command generation."""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, 
                 n_layers: int = 6, d_ff: int = 2048, max_seq_length: int = 512, 
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def get_position_embeddings(self, seq_length: int, device):
        """Generate positional embeddings."""
        positions = torch.arange(seq_length, device=device).unsqueeze(0)
        return self.position_embedding(positions)
    
    def encode(self, input_ids, attention_mask=None):
        """Encode input sequence."""
        seq_length = input_ids.size(1)
        device = input_ids.device
        
        # Token embeddings + positional embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.get_position_embeddings(seq_length, device)
        x = self.dropout(token_emb + pos_emb)
        
        # Pass through encoder layers
        encoder_attention_weights = []
        for layer in self.encoder_layers:
            x, attn_weights = layer(x, attention_mask)
            encoder_attention_weights.append(attn_weights)
        
        return x, encoder_attention_weights
    
    def decode(self, target_ids, encoder_output, src_mask=None, tgt_mask=None):
        """Decode target sequence."""
        seq_length = target_ids.size(1)
        device = target_ids.device
        
        # Token embeddings + positional embeddings
        token_emb = self.token_embedding(target_ids)
        pos_emb = self.get_position_embeddings(seq_length, device)
        x = self.dropout(token_emb + pos_emb)
        
        # Pass through decoder layers
        decoder_attention_weights = []
        for layer in self.decoder_layers:
            x, attn_weights = layer(x, tgt_mask)
            decoder_attention_weights.append(attn_weights)
        
        return x, decoder_attention_weights
    
    def forward(self, input_ids, target_ids=None, attention_mask=None, target_mask=None):
        """Forward pass of the model."""
        # Encode input
        encoder_output, encoder_attention = self.encode(input_ids, attention_mask)
        
        if target_ids is not None:
            # Training mode: decode with target
            decoder_output, decoder_attention = self.decode(
                target_ids, encoder_output, attention_mask, target_mask
            )
            # Project to vocabulary
            logits = self.output_projection(decoder_output)
            return logits, encoder_attention, decoder_attention
        else:
            # Inference mode: return encoder output for generation
            return encoder_output, encoder_attention

class NL2BashModel(nn.Module):
    """Wrapper model that combines custom transformer with T5 for better performance."""
    
    def __init__(self, tokenizer_name: str = "t5-small", use_pretrained: bool = True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Add special tokens
        special_tokens = ["<CMD>", "</CMD>", "<NL>", "</NL>"]
        self.tokenizer.add_tokens(special_tokens)
        
        if use_pretrained:
            # Use pre-trained T5 and fine-tune
            self.model = T5ForConditionalGeneration.from_pretrained(tokenizer_name)
            # Resize embeddings to accommodate new tokens
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            # Use custom transformer
            config = T5Config(
                vocab_size=len(self.tokenizer),
                d_model=512,
                d_kv=64,
                d_ff=2048,
                num_layers=6,
                num_heads=8,
                dropout_rate=0.1,
                decoder_start_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            self.model = T5ForConditionalGeneration(config)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids, attention_mask=None, max_length=128, num_beams=5, 
                early_stopping=True, do_sample=False, temperature=1.0):
        """Generate bash commands from natural language."""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer

if __name__ == "__main__":
    # Test the model
    print("Testing NL2Bash Transformer Model...")
    
    model = NL2BashModel()
    tokenizer = model.get_tokenizer()
    
    # Test input
    test_input = "<NL> List all files in the current directory </NL>"
    inputs = tokenizer(test_input, return_tensors="pt", padding=True, truncation=True)
    
    print(f"Model vocab size: {len(tokenizer)}")
    print(f"Input shape: {inputs['input_ids'].shape}")
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"Output logits shape: {outputs.logits.shape}")
    
    # Test generation
    generated = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'],
        max_length=50
    )
    
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}") 