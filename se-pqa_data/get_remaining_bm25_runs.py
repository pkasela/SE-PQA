import json
import logging
import random
from os import path
import click

import tqdm

from elastic import ElasticEngine

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.WARNING
)
logger = logging.getLogger(__name__)


def load_jsonl(file: str):
    with open(file, 'r') as f:
        for lne in f:
            yield json.loads(lne)

def get_search_results_remaining(prev_run, bm25_runs, elastic_kwargs, n_results):
    search_engine = ElasticEngine(**elastic_kwargs)
    run_dict = {}
    for _id, val in tqdm.tqdm(prev_run, desc="Worker"):
        bm25_run = bm25_runs.get(_id, None)
        if bm25_run:
            run_dict[_id] = bm25_run 
        else:
            while True:
                try:
                    elastic_results = search_engine.search(' '.join(val['text'].split(' ')[:1024]), 
                                                    n_results)['hits']['hits']
                    run_dict[_id] = {hit['_id']: hit['_score'] for hit in elastic_results}    
                except Exception as e:
                    print(e)        
                    print(_id)
                    import ipdb
                    ipdb.set_trace()
                break

    return run_dict

def get_bm25_run(qrel_filepath, bm25_run_path, elastic_kwargs, n_results):
    data_gen = load_jsonl(qrel_filepath)
    train_qrels = {}
    for d in data_gen:
        qrels = {q: 1 for q in d['rel_ids']}
        train_qrels[d['id']] = {'text': d['text'], 'qrels': qrels}
    with open(bm25_run_path, 'r') as f:
        bm25_runs = json.load(f)

    dic_qrel_list = list(train_qrels.items())
    random.shuffle(dic_qrel_list)
    runs = get_search_results_remaining(dic_qrel_list, bm25_runs, elastic_kwargs, n_results)
    return runs

@click.command()
@click.option(
    "--dataset_folder",
    type=str,
    required=True,
)
@click.option(
    "--index_name",
    type=str,
    required=True,
)
@click.option(
    "--ip",
    type=str,
    default='localhost',
)
@click.option(
    "--port",
    type=int,
    default=9200,
)
@click.option(
    "--mapping_path",
    type=str,
    required=True,
)
@click.option(
    "--train_top_k",
    type=int,
    default=100,
)
@click.option(
    "--val_top_k",
    type=int,
    default=100,
)
@click.option(
    "--test_top_k",
    type=int,
    default=100,
)
def main(
    dataset_folder, 
    index_name, 
    ip, 
    port, 
    mapping_path,
    train_top_k,
    val_top_k,
    test_top_k
):
    dataset_folder = '../dataset/answer_retrieval'
    elastic_kwargs = {'name':index_name, 'ip':ip, 'port':port,
                        'indices':index_name, 'mapping':mapping_path}

    logger.warning('Getting Training bm25 runs')
    train_qrel_path = path.join(dataset_folder,'train/data.jsonl')
    train_bm25_path = path.join(dataset_folder,'train/bm25_run.json')
    train_runs = get_bm25_run(train_qrel_path, train_bm25_path, elastic_kwargs, n_results=train_top_k)
    with open(path.join(dataset_folder,'train/bm25_run.json'), 'w') as f:
        json.dump(train_runs, f)
    
    logger.warning('Getting Val bm25 runs')
    val_qrel_path = path.join(dataset_folder,'val/data.jsonl')
    val_bm25_path = path.join(dataset_folder,'val/bm25_run.json')
    val_runs = get_bm25_run(val_qrel_path, val_bm25_path, elastic_kwargs, n_results=val_top_k)
    with open(path.join(dataset_folder,'val/bm25_run.json'), 'w') as f:
        json.dump(val_runs, f)
    
    logger.warning('Getting Test bm25 runs')
    test_qrel_path = path.join(dataset_folder,'test/data.jsonl')
    test_bm25_path = path.join(dataset_folder,'test/bm25_run.json')
    test_runs = get_bm25_run(test_qrel_path, test_bm25_path, elastic_kwargs, n_results=test_top_k)
    with open(path.join(dataset_folder,'test/bm25_run.json'), 'w') as f:
        json.dump(test_runs, f)

if __name__ == "__main__":
    main()
    