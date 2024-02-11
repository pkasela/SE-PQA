import json
import logging
import multiprocessing as mp
import random
from functools import partial
from os import path
import numpy as np
import click
from ranx import Qrels, Run, evaluate

import tqdm

from elastic import ElasticEngine

logging.basicConfig(
    filename=path.join('../logs', 'optimize_bm25.log'),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.WARNING
)
logger = logging.getLogger(__name__)


def load_jsonl(file: str):
    with open(file, 'r') as f:
        for lne in f:
            yield json.loads(lne)

def upload_data(data_collection, elastic_kwargs):
    search_engine = ElasticEngine(**elastic_kwargs)

    deleted = search_engine.delete_indices(search_engine.indices)
    if deleted:
        search_engine = ElasticEngine(**elastic_kwargs)
        logger.warning(f'Previous indices {search_engine.indices} deleted!')

    with open(data_collection, 'r') as f:
        json_iterator = json.load(f)
    requests = []
    json_length = len(json_iterator)
    for i in tqdm.tqdm(json_iterator, desc='Uploading Data', total=json_length):
        upload = {'_id': i, 'text': str(json_iterator[i])}
        requests.append(upload)
    search_engine.upload(docs=requests)

def get_search_results(chunk, elastic_kwargs, n_results, verbose):
    search_engine = ElasticEngine(**elastic_kwargs)
    run_dict = {}
    colour = "red" if chunk[1] % 2 == 0 else "white"
    if verbose:
        pbar = tqdm.tqdm(chunk[0],  
                            desc=f"Worker #{chunk[1] + 1}",
                            position=chunk[1],
                            colour=colour)
    else:
        pbar = chunk[0]
    for _id, val in pbar:
        try:
            elastic_results = search_engine.search(' '.join(val['text'].split(' ')[:1024]), 
                                                    n_results)['hits']['hits']
            run_dict[_id] = {hit['_id']: hit['_score'] for hit in elastic_results}
        except Exception as e:
            print(e)        
            print(_id)
    return run_dict


def get_bm25_run(data_gen, elastic_kwargs, k1, b, CPUS, n_results, verbose):
    # data_gen = load_jsonl(qrel_filepath)
    train_qrels = {}
    only_qrels = {}
    for d in data_gen:
        qrels = {q: 1 for q in d['rel_ids']}
        only_qrels[d['id']] = qrels
        train_qrels[d['id']] = {'text': d['text'], 'qrels': qrels}
    # with open(qrel_filepath, 'r') as f:
    #     train_qrels = json.load(f)

    dic_qrel_list = list(train_qrels.items())
    # random.shuffle(dic_qrel_list)
    search_engine = ElasticEngine(**elastic_kwargs)
    search_engine.set_bm25_params(k1=k1, b=b)
    
    chunk_size = len(dic_qrel_list) // CPUS + 1
    chunked_qrel_list = [(dic_qrel_list[i:i + chunk_size], i//chunk_size) for i in range(0, len(dic_qrel_list), chunk_size)]
    get_search_results_parallel = partial(get_search_results, elastic_kwargs=elastic_kwargs, n_results=n_results, verbose=verbose)
    with mp.Pool(CPUS) as pool:
        runs = pool.map(get_search_results_parallel, chunked_qrel_list)
    
    runs_dict = {}
    for r in runs:
        for key, val in r.items():
            runs_dict[key] = val 
    return runs_dict, only_qrels

@click.command()
@click.option(
    "--dataset_folder",
    type=str,
    required=True,
)
@click.option(
    "--cpus",
    type=int,
    default=1,
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
    "--top_k",
    type=int,
    default=100,
)
@click.option(
    "--val_size",
    type=float,
    required=True,
)
@click.option(
    "--seed",
    type=int,
    default=42,
)
def main(
    dataset_folder, 
    cpus, 
    index_name, 
    ip, 
    port, 
    mapping_path,
    top_k,
    val_size,
    seed
):
    if not seed is None:
        random.seed(seed)

    elastic_kwargs = {
                        'name':index_name, 'ip':ip, 'port':port, # stack_answers
                        'indices':index_name, 'mapping':mapping_path
                    }
    logger.warning('Uploading Data to elastic server')                        
    # upload_data(data_collection=path.join(dataset_folder, 'answer_collection.json'), elastic_kwargs=elastic_kwargs)
    logger.warning('Getting Val bm25 runs')
    val_qrel_path = path.join(dataset_folder,'val/data.jsonl')
    
    data_gen = list(load_jsonl(val_qrel_path))
    
    val_size = int(len(data_gen) * val_size) if val_size <= 1 else int(val_size)
    random_val_gen = random.sample(data_gen, val_size)
    
    k1_range = np.linspace(0, 5, 20 + 1)
    b_range = np.linspace(0, 1, 20 + 1)
    
    run_scores = {}
    _ = elastic_kwargs.pop('mapping')
    best_params = ''
    best_score = 0
    for k1 in tqdm.tqdm(k1_range, desc='k1 values'):
        for b in tqdm.tqdm(b_range, desc='b values'):
            runs, qrels = get_bm25_run(random_val_gen, elastic_kwargs, k1, b, cpus, top_k, verbose=False)
            ranx_run = Run(runs)
            ranx_qrels = Qrels(qrels)
            ranx_score = evaluate(ranx_qrels, ranx_run, 'recall@100') # increase the recall!
            if ranx_score > best_score:
                best_params = f'b={b}, k1={k1}'
                best_score = ranx_score
            run_scores[f'b={b}, k1={k1}'] = ranx_score
            logger.warning(f'Config: b:{round(b, 3):5}, k1:{round(k1,3):5} -> {ranx_score}')
            
    logger.warning(json.dumps(run_scores, indent=2))
    logger.warning(f'best_params = {best_params}, score = {best_score}')    
    
if __name__ == '__main__':
    main()