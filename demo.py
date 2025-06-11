#!/usr/bin/env python3
"""
Quick demo of the NL2Bash model without full training.

This script demonstrates the model architecture and basic functionality
using a pre-trained T5 model as a starting point.
"""

import torch
from transformer_model import NL2BashModel
from data_preprocessing import get_data_statistics

def demo_model_architecture():
    """Demonstrate the model architecture and basic functionality."""
    print("🚀 NL2Bash Model Demo")
    print("=" * 50)
    
    # Show dataset stats
    print("\n📊 Dataset Statistics:")
    try:
        stats = get_data_statistics("nl2bash-data.json")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    except FileNotFoundError:
        print("  Warning: nl2bash-data.json not found")
    
    # Initialize model
    print("\n🏗 Model Architecture:")
    model = NL2BashModel(use_pretrained=True)
    tokenizer = model.get_tokenizer()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Tokenizer vocab size: {len(tokenizer)}")
    
    # Demo inputs
    demo_inputs = [
        "list all files in the current directory",
        "find files larger than 100MB",
        "count lines in all python files",
        "show running processes",
        "delete all temporary files"
    ]
    
    print("\n🔮 Model Predictions (Pre-trained T5, not fine-tuned):")
    print("Note: These are from the base T5 model, not trained on bash commands yet.")
    print("-" * 60)
    
    model.eval()
    
    for i, user_input in enumerate(demo_inputs, 1):
        print(f"\n{i}. Input: {user_input}")
        
        # Format input
        formatted_input = f"<NL> {user_input} </NL>"
        
        # Tokenize
        inputs = tokenizer(
            formatted_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Generate
        with torch.no_grad():
            try:
                generated = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=50,
                    num_beams=3,
                    early_stopping=True,
                    do_sample=False
                )
                
                # Decode
                generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                print(f"   Generated: {generated_text}")
                
            except Exception as e:
                print(f"   Error generating: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 To get meaningful bash commands, train the model using:")
    print("   python3 nl2bash_cli.py train --data_path nl2bash-data.json --num_epochs 5")
    print("\n💡 Then use for inference:")
    print("   python3 nl2bash_cli.py generate --model_path checkpoints/best_model.pt --interactive")

def demo_attention_mechanism():
    """Demonstrate the attention mechanism."""
    print("\n🔍 Attention Mechanism Demo:")
    print("-" * 30)
    
    from transformer_model import MultiHeadAttention
    import torch.nn.functional as F
    
    # Create a simple attention layer
    d_model = 512
    n_heads = 8
    attention = MultiHeadAttention(d_model, n_heads)
    
    # Dummy input (batch_size=1, seq_len=10, d_model=512)
    x = torch.randn(1, 10, d_model)
    
    # Forward pass
    output, attention_weights = attention(x, x, x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {attention_weights.shape}")
    print(f"  Number of attention heads: {n_heads}")
    print(f"  Model dimension: {d_model}")
    
    # Show attention pattern (simplified)
    avg_attention = attention_weights.mean(dim=1).squeeze()  # Average over heads
    print(f"  Average attention matrix shape: {avg_attention.shape}")
    print("  Attention focuses on input positions (higher values = more attention)")

def main():
    """Run the demo."""
    try:
        demo_model_architecture()
        demo_attention_mechanism()
        
        print("\n✅ Demo completed successfully!")
        print("\n📚 Next steps:")
        print("1. Train the model: python3 nl2bash_cli.py train --num_epochs 5")
        print("2. Test inference: python3 nl2bash_cli.py generate --model_path checkpoints/best_model.pt --interactive")
        print("3. Read the README.md for detailed instructions")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error during demo: {e}")

if __name__ == "__main__":
    main() 