import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# import wandb  # Optional dependency
from typing import Dict, List
import argparse

from data_preprocessing import create_data_loaders, get_data_statistics
from transformer_model import NL2BashModel

class NL2BashTrainer:
    """Trainer class for the NL2Bash transformer model."""
    
    def __init__(self, model: NL2BashModel, device: str = "auto", use_wandb: bool = False):
        self.model = model
        self.device = self._get_device(device)
        self.model.to(self.device)
        self.tokenizer = model.get_tokenizer()
        self.use_wandb = use_wandb
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        if use_wandb:
            try:
                import wandb
                wandb.init(project="nl2bash-transformer")
            except ImportError:
                print("Warning: wandb not installed, skipping W&B logging")
                self.use_wandb = False
    
    def _get_device(self, device: str) -> str:
        """Get the appropriate device for training."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def compute_loss(self, outputs, labels, attention_mask=None):
        """Compute loss with proper masking."""
        logits = outputs.logits
        
        # Shift labels for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss computation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Ignore padding tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits, shift_labels)
        
        return loss
    
    def evaluate_batch(self, batch):
        """Evaluate a single batch and return metrics."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss if hasattr(outputs, 'loss') else self.compute_loss(outputs, labels)
            
            # Generate predictions for accuracy calculation
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=labels.size(1),
                num_beams=1,
                do_sample=False
            )
            
            # Compute token-level accuracy
            generated_flat = generated.view(-1)
            labels_flat = labels.view(-1)
            
            # Mask out padding tokens
            mask = labels_flat != self.tokenizer.pad_token_id
            if mask.sum() > 0:
                accuracy = (generated_flat[mask] == labels_flat[mask]).float().mean().item()
            else:
                accuracy = 0.0
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'generated': generated,
            'labels': labels
        }
    
    def validate(self, val_loader):
        """Run validation on the entire validation set."""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        generated_examples = []
        
        for batch in tqdm(val_loader, desc="Validating"):
            metrics = self.evaluate_batch(batch)
            
            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
            num_batches += 1
            
            # Store some examples
            if len(generated_examples) < 5:
                for i in range(min(2, len(batch['invocation']))):
                    generated_text = self.tokenizer.decode(
                        metrics['generated'][i], 
                        skip_special_tokens=True
                    )
                    target_text = batch['cmd'][i]
                    
                    generated_examples.append({
                        'input': batch['invocation'][i],
                        'generated': generated_text,
                        'target': target_text
                    })
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'examples': generated_examples
        }
    
    def train_epoch(self, train_loader, optimizer, scheduler=None):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss if hasattr(outputs, 'loss') else self.compute_loss(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        'train_loss_step': loss.item(),
                        'learning_rate': optimizer.param_groups[0]['lr']
                    })
                except ImportError:
                    pass
        
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader, num_epochs: int = 10, 
              learning_rate: float = 5e-5, weight_decay: float = 0.01,
              warmup_steps: int = 1000, save_dir: str = "checkpoints"):
        """Main training loop."""
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        best_val_loss = float('inf')
        
        print(f"Training for {num_epochs} epochs on {self.device}")
        print(f"Total training steps: {total_steps}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            self.train_losses.append(train_loss)
            self.learning_rates.append(optimizer.param_groups[0]['lr'])
            
            # Validation
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['loss']
            val_accuracy = val_metrics['accuracy']
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            
            # Log examples
            print("\nGenerated Examples:")
            for i, example in enumerate(val_metrics['examples'][:3]):
                print(f"  Example {i+1}:")
                print(f"    Input: {example['input']}")
                print(f"    Generated: {example['generated']}")
                print(f"    Target: {example['target']}")
                print()
            
            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_accuracy': val_accuracy
                    })
                except ImportError:
                    pass
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(os.path.join(save_dir, "best_model.pt"))
                print(f"Saved best model with val_loss: {val_loss:.4f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
                self.save_checkpoint(checkpoint_path, epoch, optimizer, scheduler)
        
        # Save final model
        self.save_model(os.path.join(save_dir, "final_model.pt"))
        self.plot_training_curves(save_dir)
        
        print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    
    def save_model(self, path: str):
        """Save the model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
        }, path)
    
    def save_checkpoint(self, path: str, epoch: int, optimizer, scheduler):
        """Save a complete training checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'tokenizer': self.tokenizer,
        }, path)
    
    def plot_training_curves(self, save_dir: str):
        """Plot and save training curves."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, label='Train Loss')
        ax1.plot(epochs, self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Learning rate
        ax2.plot(epochs, self.learning_rates)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True)
        
        # Loss difference
        loss_diff = [abs(t - v) for t, v in zip(self.train_losses, self.val_losses)]
        ax3.plot(epochs, loss_diff)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('|Train Loss - Val Loss|')
        ax3.set_title('Overfitting Monitor')
        ax3.grid(True)
        
        # Smoothed losses
        if len(self.train_losses) > 3:
            smooth_train = np.convolve(self.train_losses, np.ones(3)/3, mode='valid')
            smooth_val = np.convolve(self.val_losses, np.ones(3)/3, mode='valid')
            smooth_epochs = range(2, len(smooth_train) + 2)
            ax4.plot(smooth_epochs, smooth_train, label='Smoothed Train Loss')
            ax4.plot(smooth_epochs, smooth_val, label='Smoothed Val Loss')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_title('Smoothed Loss Curves')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train NL2Bash Transformer Model')
    parser.add_argument('--data_path', type=str, default='nl2bash-data.json',
                       help='Path to the training data')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Print dataset statistics
    print("Dataset Statistics:")
    stats = get_data_statistics(args.data_path)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, tokenizer = create_data_loaders(
        args.data_path, 
        batch_size=args.batch_size
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = NL2BashModel(use_pretrained=True)
    
    # Create trainer
    trainer = NL2BashTrainer(model, device=args.device, use_wandb=args.use_wandb)
    
    # Start training
    print(f"\nStarting training on {trainer.device}...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main() 