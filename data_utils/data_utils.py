import time
import pandas as pd
from transformers import BatchEncoding


def get_data_from_csv(logger, csv_file, fields, data_type):
    logger.info("Getting data from {} data csv file:{}".format(data_type, csv_file))
    data = pd.read_csv(csv_file)
    data.sort_values(by=fields['order_by_fields'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def get_data_from_pkl(logger, pkl_file, fields, data_type):
    logger.info("Getting data from {} data pickle file:{}".format(data_type, pkl_file))
    data = pd.read_pickle(pkl_file)
    data.sort_values(by=fields['order_by_fields'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def append_insep(data, tokenizer, message_column):
    data[message_column] = data[message_column] + tokenizer.sep_token

def concat(data, message_column, user_id_column):
    return data.groupby(user_id_column)[message_column].apply(''.join).reset_index()

def process_data(data, tokenizer, block_size, message_column):

    def tokenize(data):
        return tokenizer(data)
    
    def pad(data, pad_value):
        multiplier = (block_size - len(data))%block_size
        data.extend([pad_value]*multiplier)
        return data
    
    def convert_to_int(data):
        data['input_ids'] = list(map(int,data['input_ids']))
        data['attention_mask'] = list(map(int,data['attention_mask']))
        data['labels'] = list(map(int,data['labels']))
    
    def chunks(data):
        i_values = data['input_ids']
        a_values = data['attention_mask']
        l_values = data['labels']
        return [BatchEncoding(dict(input_ids = i_values[x:x+block_size], 
                            attention_mask=a_values[x:x+block_size], labels = l_values[x:x+block_size])) 
                            for x in range(0, len(i_values), block_size)]

    def process(data):
        tokenized = tokenize(data)
        tokenized['labels'] = tokenized['input_ids'].copy()
        tokenized['input_ids'] = pad(tokenized['input_ids'], tokenizer.eos_token_id)
        tokenized['attention_mask'] = pad(tokenized['attention_mask'], 0)
        tokenized['labels'] = pad(tokenized['labels'], -100)
        convert_to_int(tokenized)
        return chunks(tokenized)

    data['batch_encodings'] = data[message_column].apply(process)

def process_data_inference(data, tokenizer, block_size, message_column):
    """ process data when there's no label"""
    def tokenize(data):
        return tokenizer(data)
    
    def pad(data, pad_value):
        multiplier = (block_size - len(data))%block_size
        data.extend([pad_value]*multiplier)
        return data
    
    def convert_to_int(data):
        data['input_ids'] = list(map(int,data['input_ids']))
        data['attention_mask'] = list(map(int,data['attention_mask']))
        
    def chunks(data):
        i_values = data['input_ids']
        a_values = data['attention_mask']
        return [BatchEncoding(dict(input_ids = i_values[x:x+block_size], 
                            attention_mask=a_values[x:x+block_size])) 
                            for x in range(0, len(i_values), block_size)]
    
    def process(data):
        tokenized = tokenize(data)
        tokenized['input_ids'] = pad(tokenized['input_ids'], tokenizer.eos_token_id)
        tokenized['attention_mask'] = pad(tokenized['attention_mask'], 0)
        convert_to_int(tokenized)
        return chunks(tokenized)
    
    data['batch_encodings'] = data[message_column].apply(process) 
   
def transform_data(logger, tokenizer, data, block_size, message_column, user_id_column, inference = False):
    start_time = time.time()
    if user_id_column not in data.columns:
        data[user_id_column] = range(len(data))
    data_new = data[[user_id_column, message_column]].copy()
    append_insep(data_new, tokenizer, message_column)
    data_new = concat(data_new, message_column, user_id_column)
    if inference:
        process_data_inference(data_new, tokenizer, block_size, message_column)
    else:
        process_data(data_new, tokenizer, block_size, message_column)
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    return data_new 
    
def group_data(data, max_blocks, logger):
    batch = pd.DataFrame(data.batch_encodings.tolist())
    actual_blocks = len(batch.columns)
    logger.info('************** Total Number of blocks = {} *************'.format(len(batch.columns)))
    if max_blocks is not None and len(batch.columns) > max_blocks:
        batch = batch[range(max_blocks)]
        logger.info('************ Trimmed Number of blocks = {} *************'.format(len(batch.columns)))
    return batch.to_numpy().tolist(), actual_blocks

def load_dataset(logger, tokenizer, table, block_size, max_blocks, data_type, message_column, user_id_column):
    fields = {
            'order_by_fields': [user_id_column] #TODO - make this configurable
    }
    if 'pkl' in table:
        data = get_data_from_pkl(logger, table, fields, data_type)
    elif 'csv' in table:
        data = get_data_from_csv(logger, table, fields, data_type)
    data = transform_data(logger, tokenizer, data, block_size, message_column, user_id_column)
    logger.info('************** Block size = {} *************'.format(block_size))
    return group_data(data, max_blocks, logger) 

def load_dataset_from_dataframe(logger, tokenizer, data, block_size, max_blocks, message_column, user_id_column, inference = True):
    data = transform_data(logger, tokenizer, data, block_size, message_column, user_id_column, inference)
    logger.info('************** Block size = {} *************'.format(block_size))
    return group_data(data, max_blocks, logger)