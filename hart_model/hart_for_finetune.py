import torch
from hart_model.hart_for_user_embeddings import HaRTForUserEmbeddings


class HaRTforAuthorshipAttr(torch.nn.Module):

    def __init__(self,model_name_or_path=None, pt_model=None, config = None):
        if model_name_or_path:
            self.transformer = HaRTForUserEmbeddings.from_pretrained(config, model_name_or_path)
        elif pt_model:
            self.transformer = pt_model
        else:
            self.transformer = HaRTForUserEmbeddings(config=config)
            self.init_weights()
                
    def forward(
        self,
        history_input_ids = None,
        msg1_input_ids = None,
        msg2_input_ids = None,
        msg3_input_ids = None,
        history_attention_mask = None,
        msg1_attention_mask = None,
        msg2_attention_mask = None,
        msg3_attention_mask = None,
    ):

        history_rep = self.transformer(history_input_ids, history_attention_mask, history = None)

        user_rep_1 = self.transformer(msg1_input_ids, msg1_attention_mask,history_rep)
        
        user_rep_2 = self.transformer(msg2_input_ids, msg2_attention_mask, history_rep)

        user_rep_3 = self.transformer(msg3_input_ids, msg3_attention_mask, history_rep)

        
        dist_pos = torch.norm(user_rep_1 - user_rep_2)
        dist_neg = torch.norm(user_rep_1 - user_rep_3)
        loss = torch.max(torch.tensor(0.0), dist_pos - dist_neg + 1) #TODO: tune margin
        return loss
