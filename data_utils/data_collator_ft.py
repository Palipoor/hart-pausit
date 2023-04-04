import torch
from typing import Dict, List

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

@dataclass
class DataCollatorForFineTuning:

    def __init__(self, model_args, config, tokenizer: PreTrainedTokenizerBase, deepspeed=False):
        self.tokenizer = tokenizer
        if model_args.add_history or config.add_history:
            self.history = torch.load(model_args.initial_history) if model_args.initial_history else (torch.zeros(config.n_embd))
            self.history = self.history.to(torch.float16) if deepspeed else self.history.float() 
        else:
            self.history = None 
            self.layer_ins = None
            self.extract_layer = None       

    def __call__(self, examples: List[List[Dict[str, List]]]) -> List[Dict[str, torch.Tensor]]:
        first = examples[0][0]
        batch = {}

        # Handling all possible keys as figured from the first element        
        for k, v in first.items():
            if v is not None and not isinstance(v, str):
                pad = self.tokenizer.eos_token_id if k.endswith('input_ids') else 0 if k.endswith('attention_mask') else -100 
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([
                                            [block[k] if block is not None 
                                                    else ([pad]*len(v)) 
                                                for block in example] 
                                            for example in examples]) 
                else:
                    # Running through each example (i.e., each user of the batch, each user will have multiple blocks of words) 
                    batch[k] = torch.tensor([
                                            [block[k] if block is not None 
                                                    else ([pad]*len(v)) 
                                                for block in example] 
                                            for example in examples
                                            ]) 
        
        block_size = len(first['input_ids'])
        batch['history'] = None if self.history is None else self.history.repeat(len(examples), block_size, 1)

        return batch 

