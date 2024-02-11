import random
from torch.utils.data import Dataset
import torch
from transformers import T5Tokenizer
import random 

class QuestionData(Dataset):
    def __init__(self, data, answer_collection):
        self.data = data
        self.answers = answer_collection
        self.data_ids = list(self.data.keys())

    def __getitem__(self, idx):
        query_id = self.data_ids[idx]
        query = self.data[query_id]
        question = str(query['text'])
        
        all_pos_docs = query['relevant_docs']
        positive_doc = random.choices(all_pos_docs, k=1)[0]
        pos_text = str(self.answers[positive_doc])
        
        
        return {'question': question, 'pos_text': pos_text}

    def __len__(self):
        return len(self.data)

    @staticmethod
    def pad_author_vector(texts, pad_size):
        if len(texts) >= pad_size:
            pad_text = random.sample(texts, pad_size)

        if len(texts) < pad_size:
            pad_dim = pad_size - len(texts)
            pad_text = texts + ['[PAD]' for _ in range(pad_dim)]

        assert len(pad_text) == pad_size, 'error in pad_author'

        return pad_text



class StarQuestionData(Dataset):
    def __init__(self, data, answer_collection, pad_size=10):
        self.data = data
        self.answers = answer_collection
        self.data_ids = list(self.data.keys())
        self.pad_size = pad_size

    def __getitem__(self, idx):
        query_id = self.data_ids[idx]
        query = self.data[query_id]
        question = str(query['text'])
        
        positive_doc = random.choices(query['relevant_docs'], k=1)[0]
        pos_text = str(self.answers[positive_doc])

        negative_doc = random.choices(query['bm25_doc_ids'], k=1)[0]
        neg_text = str(self.answers[negative_doc])
        
        return {'question': question, 'pos_text': pos_text, 'neg_text': neg_text}

    def __len__(self):
        return len(self.data_ids)

    @staticmethod
    def pad_author_vector(texts, pad_size):
        if len(texts) >= pad_size:
            pad_text = random.sample(texts, pad_size)

        if len(texts) < pad_size:
            pad_dim = pad_size - len(texts)
            pad_text = texts + ['[PAD]' for _ in range(pad_dim)]

        assert len(pad_text) == pad_size, 'error in pad_author'

        return pad_text


def in_batch_negative_collate_fn(batch):
    question_texts = [x['question'] for x in batch]
    
    pos_texts = list(enumerate(x['pos_text'] for x in batch))
    if len(pos_texts) > 1:
        neg_texts = [random.choice(pos_texts[:i] + pos_texts[i+1:])[0] for i in range(len(pos_texts))]
    else: 
        neg_texts = [-1]
    
    return {
        'question': question_texts,
        'pos_text': [x.get('pos_text') for x in batch],
        'bm25_neg_text': [x.get('neg_text', None) for x in batch],
        'neg_text': neg_texts
    }

def batch_tokenize_preprocess(batch):
    tokenizer = T5Tokenizer.from_pretrained("castorini/monot5-small-msmarco-10k")
    relevant, target = batch['Relevant'], batch["Label_Token"]
    source_question = tuple([x.split(' Document: ')[0] for x in batch['Prompt']])
    source_doc = tuple(['Document: ' + x.split(' Document: ')[1] for x in batch['Prompt']])

    source_tokenized_question = tokenizer(
        source_question, padding="max_length", truncation = True, return_tensors='pt', max_length = 256
    )

    source_tokenized_doc = tokenizer(
        source_doc, padding="max_length", truncation = True, return_tensors='pt', max_length = 256
    )

    target_tokenized = tokenizer(
        target, padding="max_length", return_tensors='pt', max_length = 2
    )

    relevant_tokenized = tokenizer(
        relevant, padding="longest", return_tensors='pt'
    )

    for key, enc_value in list(source_tokenized_question.items()):
        enc_doc = source_tokenized_doc[key]
        enc_doc[enc_doc==1] = 0
        enc_value[enc_value==1] = 0

        liv_query = enc_value.argmin(dim = 1).tolist()
        liv_query = [x if x<127 else 127 for x in liv_query]

        liv_doc = [254-x for x in liv_query]

        enc = torch.zeros(enc_value.shape[0],254)

        for i in range(enc.shape[0]):
            enc[i,:] = torch.cat([enc_value[i,:liv_query[i]],enc_doc[i,:liv_doc[i]]])

        enc_value = enc[:, :-1].long()
        source_tokenized_question[key] = torch.cat([enc_value, relevant_tokenized[key][:enc_value.shape[0]]], dim=1) 
    
    batch = {k: v for k, v in source_tokenized_question.items()}
    # Ignore padding in the loss
    label= torch.tensor([
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ])

    batch['labels'] = label

    return batch


def in_batch_negative_collate_fn_bm25_t5(batch):

    prompt_positive = ['Question: ' + x['question'] + ' Document: ' + x['pos_text'] for x in batch]
    prompt_negative = ['Question: ' + x['question'] + ' Document: ' + x.get('neg_text', None) for x in batch]
    prompt = prompt_positive + prompt_negative

    relevant = [' Relevant: ' for x in prompt]
    label_token_positive = ['true' for x in prompt_positive]
    label_token_negative = ['false' for x in prompt_negative]
    label_token = label_token_positive + label_token_negative

    zipped = list(zip(prompt, label_token))
    random.shuffle(zipped)
    prompt, label_token = zip(*zipped)
    
    batch = {
        'Prompt' : prompt,
        'Relevant': relevant,
        'Label_Token': label_token        
    }

    batch = batch_tokenize_preprocess(batch)
    
    return batch
