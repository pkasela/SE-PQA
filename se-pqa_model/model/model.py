import torch
from torch import clamp as t_clamp
from torch import nn
from torch import tensor
from torch import sum as t_sum
from torch import max as t_max
from torch.nn import functional as F

class BiEncoder(nn.Module):
    def __init__(self, doc_model, tokenizer, device, mode='mean'):
        super(BiEncoder, self).__init__()
        # self.query_model = query_model.to(device)
        self.doc_model = doc_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        assert mode in ['max', 'mean'], 'Only max and mean pooling allowed'
        self.pooling = self.mean_pooling if mode == 'mean' else self.max_pooling
        
    def query_encoder(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        embeddings = self.doc_model(**encoded_input)
        return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)

    def doc_encoder(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        embeddings = self.doc_model(**encoded_input)
        return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)
    
    def forward(self, triplet_texts):
        query_embedding = self.query_encoder(triplet_texts[0])
        pos_embedding = self.doc_encoder(triplet_texts[1])
        neg_embedding = self.doc_encoder(triplet_texts[2])
        
        return query_embedding, pos_embedding, neg_embedding

    def forward_random_neg(self, triplet):
        query_embedding = self.query_encoder(triplet[0])
        pos_embedding = self.doc_encoder(triplet[1])
        if triplet[2][0] >= 0:
            neg_embedding = pos_embedding[tensor(triplet[2])]# self.doc_encoder(triplet_texts[2])
        else:
            print('A problem with batch size')
            neg_embedding = self.doc_encoder(['SEP'])

        return query_embedding, pos_embedding, neg_embedding
        
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return t_sum(token_embeddings * input_mask_expanded, 1) / t_clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def max_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return t_max(token_embeddings, 1)[0]


class T5Ranker(nn.Module):
    def __init__(self, doc_model, tokenizer, device):
        super(T5Ranker, self).__init__()
        self.doc_model = doc_model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        self.true_token = self.tokenizer.encode('true')[0]
        self.false_token = self.tokenizer.encode('false')[0]
        
    def forward(self, triplet_texts):
        raise NotImplementedError
    
    def get_scores(self, query, documents, batch_size=100):
        
        max_source_length = 512
        prompt_texts = [f"Query: {query} Document: {d}" for d in documents]
        source, relevant  = prompt_texts, ['Relevant: ' for d in documents]
        
        source_tokenized = self.tokenizer(source, padding="max_length", truncation=True, max_length=max_source_length-1, return_tensors='pt')
        relevant_tokenized = self.tokenizer(relevant, padding="longest", return_tensors='pt')

        for key, enc_value in list(source_tokenized.items()):
            enc_value = enc_value[:, :-1] # chop off end of sequence token-- this will be added with the prompt
            enc_value = enc_value[:, :max_source_length] # truncate any tokens that will not fit once the prompt is added
            source_tokenized[key] = torch.cat([enc_value, relevant_tokenized[key][:enc_value.shape[0]]], dim=1) # add in the prompt to the end


        prompt_texts = {k: v.to(self.device) for k, v in source_tokenized.items()}
        
        final_scores = []
        
        output = self.doc_model.generate(**prompt_texts, 
                                                output_scores=True, 
                                                return_dict_in_generate=True,
                                                max_new_tokens=20)
        output_scores = output.scores[0] # First token is either true of false
        batch_scores = output_scores[:,[self.false_token, self.true_token]]
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        batch_log_probs = batch_scores[:, 1].cpu().detach().tolist()
        final_scores.extend(batch_log_probs)
        
        return final_scores
    
