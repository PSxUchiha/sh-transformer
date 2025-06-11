import torch
from transformers import AutoTokenizer
import argparse
import os
from typing import List, Dict

from transformer_model import NL2BashModel

class NL2BashInference:
    """Inference class for generating bash commands from natural language."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.device = self._get_device(device)
        self.model, self.tokenizer = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def _get_device(self, device: str) -> str:
        """Get the appropriate device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_model(self, model_path: str):
        """Load the trained model and tokenizer."""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model
        model = NL2BashModel(use_pretrained=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get tokenizer
        tokenizer = checkpoint['tokenizer']
        
        return model, tokenizer
    
    def preprocess_input(self, text: str) -> str:
        """Preprocess the input text."""
        # Clean and format the input
        text = text.strip()
        if not text.startswith("<NL>"):
            text = f"<NL> {text} </NL>"
        return text
    
    def postprocess_output(self, text: str) -> str:
        """Postprocess the generated output."""
        # Remove special tokens and clean up
        text = text.replace("<CMD>", "").replace("</CMD>", "")
        text = text.replace("<NL>", "").replace("</NL>", "")
        return text.strip()
    
    def generate_command(self, natural_language: str, 
                        max_length: int = 128, 
                        num_beams: int = 5,
                        temperature: float = 1.0,
                        do_sample: bool = False,
                        top_p: float = 0.9,
                        repetition_penalty: float = 1.1) -> Dict[str, str]:
        """Generate a bash command from natural language description."""
        
        # Preprocess input
        input_text = self.preprocess_input(natural_language)
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        command = self.postprocess_output(generated_text)
        
        return {
            'input': natural_language,
            'generated_command': command,
            'full_generated': generated_text
        }
    
    def generate_multiple(self, natural_language: str, 
                         num_returns: int = 3,
                         **kwargs) -> List[Dict[str, str]]:
        """Generate multiple bash commands for the same input."""
        
        # Preprocess input
        input_text = self.preprocess_input(natural_language)
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate multiple outputs
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=kwargs.get('max_length', 128),
                num_return_sequences=num_returns,
                num_beams=max(num_returns, kwargs.get('num_beams', 5)),
                temperature=kwargs.get('temperature', 1.0),
                do_sample=kwargs.get('do_sample', True),
                top_p=kwargs.get('top_p', 0.9),
                repetition_penalty=kwargs.get('repetition_penalty', 1.1),
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        results = []
        for i in range(num_returns):
            generated_text = self.tokenizer.decode(generated[i], skip_special_tokens=True)
            command = self.postprocess_output(generated_text)
            
            results.append({
                'input': natural_language,
                'generated_command': command,
                'full_generated': generated_text,
                'rank': i + 1
            })
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Generate bash commands from natural language')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--input', type=str,
                       help='Natural language input')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--multiple', type=int, default=1,
                       help='Generate multiple commands (default: 1)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for inference')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum generation length')
    parser.add_argument('--num_beams', type=int, default=5,
                       help='Number of beams for beam search')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for generation')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    inference = NL2BashInference(args.model_path, device=args.device)
    print(f"Model loaded on {inference.device}")
    
    if args.interactive:
        # Interactive mode
        print("\n=== NL2Bash Interactive Mode ===")
        print("Enter natural language descriptions to generate bash commands.")
        print("Type 'quit' or 'exit' to stop.")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nEnter description: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                if args.multiple > 1:
                    results = inference.generate_multiple(
                        user_input,
                        num_returns=args.multiple,
                        max_length=args.max_length,
                        num_beams=args.num_beams,
                        temperature=args.temperature,
                        do_sample=True
                    )
                    
                    print(f"\nGenerated {len(results)} commands:")
                    for result in results:
                        print(f"  {result['rank']}. {result['generated_command']}")
                else:
                    result = inference.generate_command(
                        user_input,
                        max_length=args.max_length,
                        num_beams=args.num_beams,
                        temperature=args.temperature
                    )
                    
                    print(f"\nGenerated command: {result['generated_command']}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.input:
        # Single input mode
        if args.multiple > 1:
            results = inference.generate_multiple(
                args.input,
                num_returns=args.multiple,
                max_length=args.max_length,
                num_beams=args.num_beams,
                temperature=args.temperature,
                do_sample=True
            )
            
            print(f"Input: {args.input}")
            print(f"Generated {len(results)} commands:")
            for result in results:
                print(f"  {result['rank']}. {result['generated_command']}")
        else:
            result = inference.generate_command(
                args.input,
                max_length=args.max_length,
                num_beams=args.num_beams,
                temperature=args.temperature
            )
            
            print(f"Input: {result['input']}")
            print(f"Generated command: {result['generated_command']}")
    
    else:
        print("Please provide either --input or use --interactive mode")

if __name__ == "__main__":
    main() 