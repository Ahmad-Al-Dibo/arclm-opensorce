"""Instruction-tuning datasets with response-only loss masking."""

import torch
from torch.utils.data import Dataset, DataLoader

class InstructionDataset(Dataset):
    """Dataset for instruction-following: loss only on response tokens."""
    
    def __init__(self, instructions, responses, tokenizer, block_size, 
                 instruction_prefix="<|instruction|>", response_prefix="<|response|>"):
        """
        Args:
            instructions: List of instruction strings
            responses: List of response strings
            tokenizer: Tokenizer with encode() method
            block_size: Max sequence length
            instruction_prefix: Token to mark instructions
            response_prefix: Token to mark responses
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.instruction_prefix = instruction_prefix
        self.response_prefix = response_prefix
        
        # Preprocess
        self.data = []
        for instr, resp in zip(instructions, responses):
            full_text = f"{self.instruction_prefix}\n{instr}\n{self.response_prefix}\n{resp}"
            encoded = tokenizer.encode_text(full_text)
            
            # Token-level response mask. __getitem__ shifts it to label positions.
            instr_encoded = tokenizer.encode_text(f"{self.instruction_prefix}\n{instr}\n{self.response_prefix}\n")
            instr_len = len(instr_encoded)
            
            mask = [0.0] * instr_len + [1.0] * (len(encoded) - instr_len)
            self.data.append({'tokens': encoded, 'mask': mask})
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens'][:self.block_size]
        mask = item['mask'][:self.block_size]
        
        # Pad if needed
        pad_len = self.block_size - len(tokens)
        if pad_len > 0:
            tokens = tokens + [0] * pad_len
            mask = mask + [0.0] * pad_len
        
        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor(tokens[1:] + [0], dtype=torch.long)  # Shift for next-token
        label_mask = mask[1:] + [0.0]
        mask_tensor = torch.tensor(label_mask, dtype=torch.float32)
        
        return {'x': x, 'y': y, 'mask': mask_tensor}

def create_instruction_dataloader(instructions, responses, tokenizer, 
                                  block_size, batch_size, shuffle=True):
    """Create DataLoader for instruction tuning.
    
    
    """
    dataset = InstructionDataset(instructions, responses, tokenizer, block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
