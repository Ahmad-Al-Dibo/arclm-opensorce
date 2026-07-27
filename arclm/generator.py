"""
Generator Module - Text generation utilities
"""

import torch
import math


class Generator:
    """Text generation with trained model"""
    
    def __init__(self, model, stoi, itos, block_size, device, tokenizer=None):
        self.model = model
        self.stoi = stoi
        self.itos = itos
        self.block_size = block_size
        self.device = device
        self.tokenizer = tokenizer
    
    def generate(self, start_text, max_new_tokens=20, temperature=1.0, 
                 repetition_penalty=1.0, top_k=None, top_p=None):
        """Generate text from start text
        
        Args:
            start_text: Starting text for generation
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (0.5=conservative, 2.0=wild)
            repetition_penalty: Penalty for repeating tokens (1.0=no penalty, 2.0=high penalty)
            top_k: Keep only top k tokens (None=all)
            top_p: Keep tokens with cumulative probability p (None=all)
        """
        self.model.eval()
        
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode_text"):
            idx = self.tokenizer.encode_text(start_text.lower())
        else:
            words = start_text.lower().split()
            idx = [self.stoi[w] for w in words if w in self.stoi]
        
        if not idx:
            return []
        
        idx = torch.tensor([idx], device=self.device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size:]
                logits = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(logits, idx[0], repetition_penalty)
                
                # Apply top-k filtering
                if top_k is not None:
                    logits = self._apply_top_k(logits, top_k)
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    logits = self._apply_top_p(logits, top_p)
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                idx = torch.cat([idx, next_token], dim=1)
                
        
        output = [self.itos[i.item()] for i in idx[0]]
        return output
    
    def _apply_repetition_penalty(self, logits, tokens, penalty):
        """Apply repetition penalty to logits
        
        Verlaagt waarschijnlijkheid van recent gebruikte tokens
        
        Formule:
            logits[token] = logits[token] / penalty (if token in recent history)
        """
        # Bereken penalty factor
        unique_tokens = set(tokens.tolist())
        
        for token in unique_tokens:
            if logits[0, token] < 0:
                logits[0, token] *= penalty
            else:
                logits[0, token] /= penalty
        
        return logits
    
    def _apply_top_k(self, logits, top_k):
        """Keep only top k highest probability tokens"""
        top_k_vals, top_k_indices = torch.topk(logits, top_k)
        
        # Create mask
        mask = torch.full_like(logits, float('-inf'))
        mask[0, top_k_indices[0]] = top_k_vals[0]
        
        return mask
    
    def _apply_top_p(self, logits, top_p):
        """Keep tokens with cumulative probability >= top_p (nucleus sampling)"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens with cumsum > top_p
        sorted_indices_to_remove = cumsum_probs > top_p
        sorted_indices_to_remove[0, 0] = False  # Always keep best token
        
        # Apply mask
        sorted_logits[0, sorted_indices_to_remove[0]] = float('-inf')
        
        # Unsort back
        logits = torch.zeros_like(logits)
        logits[0, sorted_indices[0]] = sorted_logits[0]
        
        return logits
    
    def generate_string(self, start_text, max_new_tokens=20, temperature=1.0,
                       repetition_penalty=1.0, top_k=None, top_p=None):
        """Generate text and return as string"""
        tokens = self.generate(start_text, max_new_tokens, temperature, 
                              repetition_penalty, top_k, top_p)
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode_string"):
            token_ids = [
                self.stoi[token]
                for token in tokens
                if token in self.stoi
            ]
            return self.tokenizer.decode_string(token_ids)
        return " ".join(tokens)
