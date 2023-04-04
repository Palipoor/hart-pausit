from data_utils.data_collator import DataCollatorWithPaddingForHaRT
from data_utils.data_utils import load_dataset_from_dataframe
from hart_model.hart_for_user_embeddings import HaRTForUserEmbeddings
import torch
import pandas as pd
import json
from attrdict import AttrDict
from tqdm import tqdm
import datasets
import logging
from transformers import AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)

tokenizer_path = 'hart_pt_twt'
model_path = 'hart_pt_twt'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
config = AutoConfig.from_pretrained(tokenizer_path)

queries = []
candidates = []
with open('/Users/zire/Desktop/hiatus-metrics/eval_samples_HRSBackground/mode_multi-genre/sample2_randomSeed1266677/TA1/multi-genre/multi-genre_TA1_queries.jsonl', 'r') as f:
    for line in f:
        queries.append(json.loads(line))
with open('/Users/zire/Desktop/hiatus-metrics/eval_samples_HRSBackground/mode_multi-genre/sample2_randomSeed1266677/TA1/multi-genre/multi-genre_TA1_candidates.jsonl', 'r') as f:
    for line in f:
        candidates.append(json.loads(line))
queries = pd.DataFrame(queries)
candidates = pd.DataFrame(candidates)

args = AttrDict()
args['output_block_last_hidden_states'] = True
args['initial_history'] = 'initial_history/initialized_history_tensor.pt'
args['add_history'] = True
args['layer_ins'] = 2
args['extract_layer'] = 11
data_collator = DataCollatorWithPaddingForHaRT(args, config, tokenizer)
q = load_dataset_from_dataframe(logger, tokenizer,queries, 64, 8, 'fullText', 'authorID', inference=True)[0]
c = load_dataset_from_dataframe(logger, tokenizer, candidates, 64, 8, 'fullText', 'authorID', inference=True)[0]
model = HaRTForUserEmbeddings(config)
features_q = []
features_c = []
with torch.no_grad():
    for x in tqdm(c):
        c_input = data_collator([x])
        output_c = model(c_input['input_ids'], c_input['attention_mask'], c_input['history'])
        features_c.append(output_c.flatten().tolist())
    for x in tqdm(q):
        q_input = data_collator([x])
        output_q = model(q_input['input_ids'], q_input['attention_mask'], q_input['history'])
        features_q.append(output_q.flatten().tolist())
queries['features'] = features_q
candidates['features'] = features_c
datadict = {'queries': datasets.Dataset.from_pandas(queries), 'candidates': datasets.Dataset.from_pandas(candidates)}
data = datasets.DatasetDict(datadict)
data.save_to_disk('features')
