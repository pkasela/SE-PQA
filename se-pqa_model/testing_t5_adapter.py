# Test T5 adapter

import json
import logging
import os

import click
import torch
import tqdm
from adapters import init
from dataloader.utils import load_test_query, seed_everything
from model.model import T5Ranker
from ranx import Qrels, Run, compare
from transformers import AutoTokenizer, T5ForConditionalGeneration

logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True

def get_t5_rank(data, model, doc_collection, bm25_runs, k=100):
    test_qrels = {}
    t5_run = {}
    for d in tqdm.tqdm(data, total=len(data)):    
        q_text = data[d]['text']
        d_qrels = {k: 1 for k in data[d]['relevant_docs']}
        test_qrels[d] = d_qrels
        
        bm25_docs = list(bm25_runs[d].keys())
        bm25_doc_texts = [doc_collection[doc_id] for doc_id in bm25_docs]
        with torch.no_grad():
            with torch.autocast(device_type='cuda'):
                t5_scores = model.get_scores(q_text, bm25_doc_texts)
        t5_run[d] = {doc_id: t5_scores[i] for i, doc_id in enumerate(bm25_docs)}            
        
    return test_qrels, t5_run

def testT5Adapter(data_folder, model_path, split, seed):
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
    tokenizer = AutoTokenizer.from_pretrained("castorini/monot5-base-msmarco-10k")
    
    
    doc_model = T5ForConditionalGeneration.from_pretrained("castorini/monot5-base-msmarco-10k")
    init(doc_model)

    #config = BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=48, non_linearity="relu")

    doc_model.load_adapter(model_path)
    doc_model.set_active_adapters('t5_qa')   

    doc_model.eval()
    model = T5Ranker(doc_model, tokenizer, device)

    #ipdb.set_trace()


    logger.debug('Loading Embedding File and id_to_index dictionary.')
    with open(os.path.join(data_folder, 'answer_collection.json'), 'r') as f:
            doc_collection = json.load(f)


    queries_file = f'{split}/data.jsonl'
    filename = os.path.join(data_folder, queries_file)
    data = load_test_query(filename)

    bm25_file = f'{split}/bm25_run.json'
    bm25_filename = os.path.join(data_folder, bm25_file)
    with open(bm25_filename, 'r') as f:
        bm25_run = json.load(f)

    test_qrels, bert_run = get_t5_rank(data, model, doc_collection, bm25_run) 
    
    qrels = Qrels(test_qrels)
   
    logging.info(f'Reporting {split} - {data_folder}')

    logger.debug('Saving berts file')
    with open(os.path.join(data_folder, f'{split}/{model_path.replace("/","_")}_rerank.json'), 'w') as f:
            json.dump(bert_run, f)

    logger.debug('Evaluating with Ranx')
    ranx_bert_run = Run(bert_run, name='T5_TRAINED')
   
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

@click.command()
@click.option(
    "--data_folder",
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
    required=True
)
def main(data_folder, model_path, split, seed):
    # data_folder = '../../answer_retrieval'
    # model_path = 't5_sepqa_new_len_model_base_epoch_4'

    # split='val'
    # seed = 42
    testT5Adapter(data_folder, model_path, split, seed)

if __name__ == '__main__':
     main()