from hart_model.hart_for_finetune import HaRTforAuthorshipAttr
from transformers import Trainer, TrainingArguments, AutoTokenizer



def main():
    model = HaRTforAuthorshipAttr(model_name_or_path = 'hart_pt_twt')
    tokenizer = AutoTokenizer.from_pretrained('hart_pt_twt')
    #todo load data
    trainer = Trainer(model=model, tokenizer=tokenizer,args = training_args, data_collator = data_collator, train_dataset = train_dataset, eval_dataset = eval_dataset)
    trainer.train()
    trainer.save_model()
    trainer.evaluate()