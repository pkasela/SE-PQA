import json
import logging
import os

import click
import torch
import tqdm
from ranx import Qrels, Run, compare
from transformers import AutoModel, AutoTokenizer

from dataloader.utils import load_test_query, seed_everything
from model.model import BiEncoder

logger = logging.getLogger(__name__)

def get_bert_rank(data, model, doc_embedding, bm25_runs, id_to_index, k=100):
    test_qrels = {}
    bert_run = {}
    index_to_id = {ind: _id for _id, ind in id_to_index.items()}
    for d in tqdm.tqdm(data, total=len(data)):
        q_text = data[d]['text']
        with torch.no_grad():
            q_embedding = model.query_encoder(q_text)#.cpu()
        d_qrels = {k: 1 for k in data[d]['relevant_docs']}
        test_qrels[d] = d_qrels
        
        bm25_docs = list(bm25_runs[d].keys())
        d_embeddings = doc_embedding[torch.tensor([int(id_to_index[x]) for x in bm25_docs])]
        bert_scores = torch.einsum('xy, ly -> x', d_embeddings, q_embedding)

        bert_run[d] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bm25_docs)}
        
    return test_qrels, bert_run

@click.command()
@click.option(
    "--data_folder",
    type=str,
    required=True
)
@click.option(
    "--output_folder",
    type=str,
    required=True
)
@click.option(
    "--bert_name",
    type=str,
    required=True
)
@click.option(
    "--model_path",
    type=str,
    required=True
)
@click.option(
    "--split",
    type=str,
    required=True
)
@click.option(
    "--seed",
    type=int,
    default=None
)
def main(data_folder, output_folder, bert_name, model_path, split, seed):
    if seed:
        seed_everything(seed)
        
    os.makedirs('../logs', exist_ok=True)
    logging_file = f"testing_{data_folder.split('/')[-1]}.log"
    logging.basicConfig(filename=os.path.join('../logs', logging_file),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO
                        )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.debug('Loading models and tokenizer.')
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    doc_model = AutoModel.from_pretrained(bert_name)

    model = BiEncoder(doc_model, tokenizer, device)
    logger.info(f'Loading model weights from {model_path}')
    if model_path:
        model_epoch = int(model_path.replace('.pt', '').split('_')[-1]) if model_path else 0
        model.load_state_dict(torch.load(model_path))

    logger.debug('Loading Embedding File and id_to_index dictionary.')
    doc_embedding = torch.load(os.path.join(output_folder, f'collection_embedding_{model_epoch}.pt')).to(device)
    with open(os.path.join(output_folder, f'id_to_index_{model_epoch}.json'), 'r') as f:
        id_to_index = json.load(f)


    queries_file = f'{split}/data.jsonl'
    filename = os.path.join(data_folder, queries_file)
    data = load_test_query(filename)

    bm25_file = f'{split}/bm25_run.json'
    bm25_filename = os.path.join(data_folder, bm25_file)
    with open(bm25_filename, 'r') as f:
        bm25_run = json.load(f)

    test_qrels, bert_run = get_bert_rank(data, model, doc_embedding, bm25_run, id_to_index) 
    qrels = Qrels(test_qrels)
   
    logging.info(f'Reporting {split} - {data_folder}')

    logger.debug('Saving berts file')
    with open(os.path.join(data_folder, f'{split}/bert_run_{model_epoch}_rerank.json'), 'w') as f:
        json.dump(bert_run, f)

    logger.debug('Evaluating with Ranx')
    ranx_bert_run = Run(bert_run, name='BERT')
    ranx_bm25_run = Run(bm25_run, name='BM25')

    models = [ranx_bert_run, ranx_bm25_run]
    
    report = compare(
        qrels=qrels,
        runs=models,
        metrics=['mrr@10', 'ndcg@010', 'recall@10', 'precision@10', 'map@100', 'recall@100'],
        max_p=0.01  # P-value threshold
    )
    logger.info(f'\nReRank BM25\n{report}')
    print(report)

if __name__ == '__main__':
    main()