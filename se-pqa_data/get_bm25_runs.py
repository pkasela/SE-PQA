import json
import logging
import multiprocessing as mp
import random
from functools import partial
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

def upload_data(data_collection, elastic_kwargs):
    search_engine = ElasticEngine(**elastic_kwargs)

    deleted = search_engine.delete_indices(search_engine.indices)
    if deleted:
        logger.warning(f'Previous indices {search_engine.indices} deleted!')
        search_engine = ElasticEngine(**elastic_kwargs)

    with open(data_collection, 'r') as f:
        json_iterator = json.load(f)
    requests = []
    json_length = len(json_iterator)
    for i in tqdm.tqdm(json_iterator, desc='Uploading Data', total=json_length):
        upload = {'_id': i, 'text': str(json_iterator[i])}
        requests.append(upload)
    search_engine.upload(docs=requests)

def get_search_results(chunk, elastic_kwargs, n_results):
    search_engine = ElasticEngine(**elastic_kwargs)
    run_dict = {}
    colour = "red" if chunk[1] % 2 == 0 else "white"
    for _id, val in tqdm.tqdm(chunk[0],  
                            desc=f"Worker #{chunk[1] + 1}",
                            position=chunk[1],
                            colour=colour):
        try:
            elastic_results = search_engine.search(' '.join(val['text'].split(' ')[:1024]), 
                                                    n_results)['hits']['hits']
            run_dict[_id] = {hit['_id']: hit['_score'] for hit in elastic_results}
        except Exception as e:
            print(e)        
            print(_id)
    return run_dict


def get_bm25_run(qrel_filepath, elastic_kwargs, CPUS, n_results):
    data_gen = load_jsonl(qrel_filepath)
    train_qrels = {}
    for d in data_gen:
        qrels = {q: 1 for q in d['rel_ids']}
        train_qrels[d['id']] = {'text': d['text'], 'qrels': qrels}
    # with open(qrel_filepath, 'r') as f:
    #     train_qrels = json.load(f)

    dic_qrel_list = list(train_qrels.items())
    random.shuffle(dic_qrel_list)
    chunk_size = len(dic_qrel_list) // CPUS + 1
    chunked_qrel_list = [(dic_qrel_list[i:i + chunk_size], i//chunk_size) for i in range(0, len(dic_qrel_list), chunk_size)]
    get_search_results_parallel = partial(get_search_results, elastic_kwargs=elastic_kwargs, n_results=n_results)
    with mp.Pool(CPUS) as pool:
        runs = pool.map(get_search_results_parallel, chunked_qrel_list)
    
    runs_dict = {}
    for r in runs:
        for key, val in r.items():
            runs_dict[key] = val 
    return runs_dict

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
    cpus, 
    index_name, 
    ip, 
    port, 
    mapping_path,
    train_top_k,
    val_top_k,
    test_top_k
):
    # dataset_folder = '../dataset/answer_retrieval'
    CPUS = cpus
    elastic_kwargs = {
                        'name':index_name, 'ip':ip, 'port':port, # stack_answers
                        'indices':index_name, 'mapping':mapping_path
                    }
    logger.warning('Uploading Data to elastic server')                        
    upload_data(data_collection=path.join(dataset_folder, 'answer_collection.json'), elastic_kwargs=elastic_kwargs)

    logger.warning('Getting Training bm25 runs')
    train_qrel_path = path.join(dataset_folder,'train/data.jsonl')
    train_runs = get_bm25_run(train_qrel_path, elastic_kwargs, CPUS, n_results=train_top_k)
    with open(path.join(dataset_folder,'train/bm25_run.json'), 'w') as f:
        json.dump(train_runs, f)
    
    logger.warning('Getting Val bm25 runs')
    val_qrel_path = path.join(dataset_folder,'val/data.jsonl')
    val_runs = get_bm25_run(val_qrel_path, elastic_kwargs, CPUS, n_results=val_top_k)
    with open(path.join(dataset_folder,'val/bm25_run.json'), 'w') as f:
        json.dump(val_runs, f)
    
    logger.warning('Getting Test bm25 runs')
    test_qrel_path = path.join(dataset_folder,'test/data.jsonl')
    test_runs = get_bm25_run(test_qrel_path, elastic_kwargs, CPUS, n_results=test_top_k)
    with open(path.join(dataset_folder,'test/bm25_run.json'), 'w') as f:
        json.dump(test_runs, f)
    


if __name__ == '__main__':
    main()