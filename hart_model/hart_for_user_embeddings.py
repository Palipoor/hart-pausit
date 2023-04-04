import torch
from .hart import HaRTPreTrainedModel

class HaRTForUserEmbeddings(HaRTPreTrainedModel):
        def __init__(self, config, model_name_or_path=None):        
                config.add_history = True
                super().__init__(config)
                self.add_history = True
                self.transformer = HaRTPreTrainedModel(config, model_name_or_path)
        def forward(self,
        input_ids = None,
        attention_mask = None,
        history = None):
                if history:
                        history = history.float()
                messages_transformer_outputs = self.transformer(
                        input_ids=input_ids,
                        history=history,
                        output_block_last_hidden_states=True,
                        output_block_extract_layer_hs=True,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        output_attentions=True)
                
                states = messages_transformer_outputs.history[0]
                masks = messages_transformer_outputs.history[1]
                multiplied = tuple(l * r for l, r in zip(states, masks))
                all_blocks_user_states = torch.stack(multiplied, dim=1)
                all_blocks_masks = torch.stack(masks, dim=1)
                sum = torch.sum(all_blocks_user_states, dim=1)
                divisor = torch.sum(all_blocks_masks, dim=1)
                user_representation = sum/divisor
                user_representation = user_representation.unsqueeze(1)
                return user_representation
