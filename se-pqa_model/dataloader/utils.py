import json
import logging
import os
import random
import subprocess

import numpy as np
import torch
import tqdm

logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    logger.info(f'Setting global random seed to {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


def load_jsonl(file: str):
    with open(file, 'r') as f:
        for lne in f:
            yield json.loads(lne)


def load_query_data(file: str, verbose: bool=True):
    with open(file, 'r') as f:
        query_file = {}
        pbar = tqdm.tqdm(f, desc='Creating data for loading') if verbose else f
        for lne in pbar:
            query_json = json.loads(lne)
            bm25_not_relevant_docs = [id_ for id_ in query_json['bm25_doc_ids'] if id_ not in query_json['rel_ids']]
            if bm25_not_relevant_docs:
                query_file[query_json['id']] = {
                    'text': query_json['text'],
                    'title': query_json['title'],
                    'relevant_docs': query_json['rel_ids'],
                    'bm25_doc_ids': bm25_not_relevant_docs
                }

        return query_file


def load_test_query(file):
    with open(file, 'r') as f:
        query_file = {}
        for lne in tqdm.tqdm(f):
            query_json = json.loads(lne)
            query_file[query_json['id']] = {
                'text': query_json['text'],
                'title': query_json['title'],
                'relevant_docs': query_json['rel_ids'],
                'user_questions': query_json['user_questions'],
                'user_answers': query_json['user_answers'],
                'best_answer': query_json['best_answer'],
            }
            
        return query_file