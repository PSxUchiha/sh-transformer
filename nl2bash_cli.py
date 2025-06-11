#!/usr/bin/env python3
"""
NL2Bash: Natural Language to Bash Command Generator

A transformer-based model that learns to generate bash commands from natural language descriptions.
Uses attention mechanisms and pre-trained T5 architecture for sequence-to-sequence learning.
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="NL2Bash: Natural Language to Bash Command Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python nl2bash_cli.py train --data_path nl2bash-data.json --num_epochs 5

  # Generate a single command
  python nl2bash_cli.py generate --model_path checkpoints/best_model.pt --input "list all files"

  # Interactive mode
  python nl2bash_cli.py generate --model_path checkpoints/best_model.pt --interactive

  # Generate multiple alternatives
  python nl2bash_cli.py generate --model_path checkpoints/best_model.pt --input "find large files" --multiple 3

  # Test the data preprocessing
  python nl2bash_cli.py test-data --data_path nl2bash-data.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the NL2Bash model')
    train_parser.add_argument('--data_path', type=str, default='nl2bash-data.json',
                             help='Path to the training data (default: nl2bash-data.json)')
    train_parser.add_argument('--batch_size', type=int, default=16,
                             help='Training batch size (default: 16)')
    train_parser.add_argument('--num_epochs', type=int, default=10,
                             help='Number of training epochs (default: 10)')
    train_parser.add_argument('--learning_rate', type=float, default=5e-5,
                             help='Learning rate (default: 5e-5)')
    train_parser.add_argument('--device', type=str, default='auto',
                             help='Device to use for training (auto/cpu/cuda/mps)')
    train_parser.add_argument('--use_wandb', action='store_true',
                             help='Use Weights & Biases for logging')
    train_parser.add_argument('--save_dir', type=str, default='checkpoints',
                             help='Directory to save model checkpoints (default: checkpoints)')
    
    # Generation command
    generate_parser = subparsers.add_parser('generate', help='Generate bash commands')
    generate_parser.add_argument('--model_path', type=str, required=True,
                                help='Path to the trained model checkpoint')
    generate_parser.add_argument('--input', type=str,
                                help='Natural language input')
    generate_parser.add_argument('--interactive', action='store_true',
                                help='Run in interactive mode')
    generate_parser.add_argument('--multiple', type=int, default=1,
                                help='Generate multiple commands (default: 1)')
    generate_parser.add_argument('--device', type=str, default='auto',
                                help='Device to use for inference (auto/cpu/cuda/mps)')
    generate_parser.add_argument('--max_length', type=int, default=128,
                                help='Maximum generation length (default: 128)')
    generate_parser.add_argument('--num_beams', type=int, default=5,
                                help='Number of beams for beam search (default: 5)')
    generate_parser.add_argument('--temperature', type=float, default=1.0,
                                help='Temperature for generation (default: 1.0)')
    
    # Test data command
    test_parser = subparsers.add_parser('test-data', help='Test data preprocessing')
    test_parser.add_argument('--data_path', type=str, default='nl2bash-data.json',
                            help='Path to the data file (default: nl2bash-data.json)')
    test_parser.add_argument('--batch_size', type=int, default=4,
                            help='Batch size for testing (default: 4)')
    
    # Model info command
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.add_argument('--model_path', type=str, required=True,
                            help='Path to the model checkpoint')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Import and run training
        try:
            import sys
            from train import main as train_main
            
            # Modify sys.argv to pass arguments to train.py
            original_argv = sys.argv
            sys.argv = ['train.py']
            
            if args.data_path != 'nl2bash-data.json':
                sys.argv.extend(['--data_path', args.data_path])
            if args.batch_size != 16:
                sys.argv.extend(['--batch_size', str(args.batch_size)])
            if args.num_epochs != 10:
                sys.argv.extend(['--num_epochs', str(args.num_epochs)])
            if args.learning_rate != 5e-5:
                sys.argv.extend(['--learning_rate', str(args.learning_rate)])
            if args.device != 'auto':
                sys.argv.extend(['--device', args.device])
            if args.use_wandb:
                sys.argv.append('--use_wandb')
            if args.save_dir != 'checkpoints':
                sys.argv.extend(['--save_dir', args.save_dir])
            
            train_main()
            sys.argv = original_argv
            
        except ImportError as e:
            print(f"Error importing training modules: {e}")
            print("Make sure all dependencies are installed: pip install -r requirements.txt")
            sys.exit(1)
    
    elif args.command == 'generate':
        # Import and run inference
        try:
            import sys
            from inference import main as inference_main
            
            # Check if model exists
            if not os.path.exists(args.model_path):
                print(f"Error: Model file '{args.model_path}' not found.")
                print("Please train a model first using: python nl2bash_cli.py train")
                sys.exit(1)
            
            # Modify sys.argv to pass arguments to inference.py
            original_argv = sys.argv
            sys.argv = ['inference.py', '--model_path', args.model_path]
            
            if args.input:
                sys.argv.extend(['--input', args.input])
            if args.interactive:
                sys.argv.append('--interactive')
            if args.multiple != 1:
                sys.argv.extend(['--multiple', str(args.multiple)])
            if args.device != 'auto':
                sys.argv.extend(['--device', args.device])
            if args.max_length != 128:
                sys.argv.extend(['--max_length', str(args.max_length)])
            if args.num_beams != 5:
                sys.argv.extend(['--num_beams', str(args.num_beams)])
            if args.temperature != 1.0:
                sys.argv.extend(['--temperature', str(args.temperature)])
            
            inference_main()
            sys.argv = original_argv
            
        except ImportError as e:
            print(f"Error importing inference modules: {e}")
            print("Make sure all dependencies are installed: pip install -r requirements.txt")
            sys.exit(1)
    
    elif args.command == 'test-data':
        # Test data preprocessing
        try:
            from data_preprocessing import get_data_statistics, create_data_loaders
            
            print("Testing data preprocessing...")
            
            # Check if data file exists
            if not os.path.exists(args.data_path):
                print(f"Error: Data file '{args.data_path}' not found.")
                sys.exit(1)
            
            # Get statistics
            stats = get_data_statistics(args.data_path)
            print("\nDataset Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            # Test data loader
            print(f"\nTesting data loaders with batch size {args.batch_size}...")
            train_loader, val_loader, tokenizer = create_data_loaders(
                args.data_path, 
                batch_size=args.batch_size
            )
            
            print(f"Train samples: {len(train_loader.dataset)}")
            print(f"Validation samples: {len(val_loader.dataset)}")
            print(f"Tokenizer vocab size: {len(tokenizer)}")
            
            # Show a sample
            sample_batch = next(iter(train_loader))
            print(f"\nSample batch shapes:")
            print(f"  Input IDs: {sample_batch['input_ids'].shape}")
            print(f"  Labels: {sample_batch['labels'].shape}")
            print(f"\nSample invocation: {sample_batch['invocation'][0]}")
            print(f"Sample command: {sample_batch['cmd'][0]}")
            
            print("\n✅ Data preprocessing test completed successfully!")
            
        except ImportError as e:
            print(f"Error importing data modules: {e}")
            print("Make sure all dependencies are installed: pip install -r requirements.txt")
            sys.exit(1)
        except Exception as e:
            print(f"Error during data testing: {e}")
            sys.exit(1)
    
    elif args.command == 'info':
        # Show model information
        try:
            import torch
            
            # Check if model exists
            if not os.path.exists(args.model_path):
                print(f"Error: Model file '{args.model_path}' not found.")
                sys.exit(1)
            
            # Load checkpoint
            checkpoint = torch.load(args.model_path, map_location='cpu')
            
            print(f"Model Information for: {args.model_path}")
            print("=" * 50)
            
            # Model state info
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                total_params = sum(p.numel() for p in state_dict.values())
                print(f"Total parameters: {total_params:,}")
            
            # Training info
            if 'epoch' in checkpoint:
                print(f"Trained epochs: {checkpoint['epoch'] + 1}")
            
            if 'train_losses' in checkpoint:
                train_losses = checkpoint['train_losses']
                print(f"Final training loss: {train_losses[-1]:.4f}")
            
            if 'val_losses' in checkpoint:
                val_losses = checkpoint['val_losses']
                print(f"Final validation loss: {val_losses[-1]:.4f}")
                print(f"Best validation loss: {min(val_losses):.4f}")
            
            # Tokenizer info
            if 'tokenizer' in checkpoint:
                tokenizer = checkpoint['tokenizer']
                print(f"Tokenizer vocab size: {len(tokenizer)}")
            
            print("=" * 50)
            
        except Exception as e:
            print(f"Error loading model info: {e}")
            sys.exit(1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()