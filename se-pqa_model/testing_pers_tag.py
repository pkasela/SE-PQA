import json
import logging
import os
from functools import partial
from multiprocessing import Pool

import click
import pandas as pd
import tqdm
from dataloader.utils import load_test_query, seed_everything
from ranx import Qrels, Run, compare

logger = logging.getLogger(__name__)


def create_user_tags(data, df_question, df_answers, bm25_docs, CPUS):

    df_answers = df_answers.dropna(subset='AccountId')
    df_answers.AccountId = df_answers.AccountId.astype(int)
    answer_to_id = df_answers[['Id', 'AccountId']].set_index('Id').to_dict()['AccountId']

    question_to_date = df_question[['Id', 'CreationDate']].set_index('Id').to_dict()['CreationDate']
    df_question_na = df_question.dropna(subset='AccountId')
    df_question_na.AccountId = df_question_na.AccountId.astype(int)
    question_to_id = df_question_na[['Id', 'AccountId']].set_index('Id').to_dict()['AccountId']

    df_ans_question = df_answers[['Id',  'ParentId', 
                                  'AccountId', 'CreationDate']].merge(df_question[[
                                                    'Id', 'CreationDate', 
                                                    'AccountId', 'Tags'
                                                    ]], 
                                            left_on='ParentId', right_on='Id')
    answer_to_date = df_ans_question[['Id_x', 'CreationDate_y']].set_index('Id_x').to_dict()['CreationDate_y'] # answer_to_date is the question time of the answer

    question_tags = {}
    for d in tqdm.tqdm(data, desc='Creating Question Tag Set'):
        q_date = question_to_date[d]
        asker_id = question_to_id.get(d, -999)

        asker_tags = df_question_na[(df_question_na.CreationDate <= q_date) & 
                                 (df_question_na.AccountId == asker_id)]['Tags'].sum()

        if asker_tags == 0:
            question_tags[d] = []
        else:
            question_tags[d] = set(asker_tags.strip('>').strip('<').split('><'))

    only_bm25_docs = [list(doc_list.keys()) for doc_list in bm25_docs.values()]
    all_bm25_docs = [item for sublist in only_bm25_docs for item in sublist]
    all_bm25_docs = list(set(all_bm25_docs))
    CPUS = 15
    bm25_docs_splits = len(all_bm25_docs) // CPUS
    if (len(all_bm25_docs) % CPUS) != 0:
        bm25_docs_splits += 1
    split_bm25_docs = [all_bm25_docs[i*bm25_docs_splits:(i+1)*bm25_docs_splits] for i in range(CPUS)]
    parallel_answer_tags = partial(get_answer_user_tags, answer_to_id=answer_to_id, answer_to_date=answer_to_date, df_ans_question=df_ans_question)

    with Pool(CPUS) as pool:
        res = pool.map(parallel_answer_tags, split_bm25_docs)

    answer_tags = {}
    for r in res:
        for doc in r:
            answer_tags[doc] = r[doc]
    return question_tags, answer_tags

def get_answer_user_tags(some_bm25_docs, answer_to_id, answer_to_date, df_ans_question):
    answer_tags = {}
    test_date = int(pd.to_datetime('2020-12-31 23:59:59').timestamp())
    for doc in tqdm.tqdm(some_bm25_docs, desc='Creating Answer Tags Set', miniters=1000):
        answerer_id = answer_to_id.get(doc, -999)
        a_date = answer_to_date.get(doc, test_date) # day of question of the answered considered
        answerer_tags = df_ans_question[(df_ans_question.CreationDate_x < a_date) & 
                                        (df_ans_question.AccountId_x == answerer_id)]['Tags'].sum()

        if answerer_tags == 0:
            answer_tags[doc] = []
        else:
            answer_tags[doc] = set(answerer_tags.strip('>').strip('<').split('><'))

    return answer_tags


def get_tag_rank(data, question_tags, answer_tags, bm25_runs, k=100):
    def get_score(answer_id):
        a_tags = set(answer_tags[answer_id])
        return len(q_tags & a_tags) / (len(q_tags) + 1)

    test_qrels = {}
    tag_run = {}
    for d in tqdm.tqdm(data, total=len(data)):
        d_qrels = {k: 1 for k in data[d]['relevant_docs']}
        test_qrels[d] = d_qrels

        q_tags = set(question_tags[d])

        bm25_docs = list(bm25_runs[d].keys())
        tag_scores = [get_score(x) for x in bm25_docs]

        tag_run[d] = {doc_id: tag_scores[i] for i, doc_id in enumerate(bm25_docs)}
        
    return test_qrels, tag_run

@click.command()
@click.option(
   "--df_folder",
    type=str,
    required=True
)
@click.option(
    "--data_folder",
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
@click.option(
    "--cpus",
    type=int,
    default=1
)
def main(df_folder, data_folder, split, seed, cpus):
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

    df_question = pd.read_csv(os.path.join(df_folder, 'questions.csv'), lineterminator='\n')
    df_question.CreationDate = pd.to_datetime(df_question.CreationDate).apply(lambda x: int(x.timestamp())) 
        
    df_answers = pd.read_csv(os.path.join(df_folder, 'answers.csv'), lineterminator='\n')
    df_answers.CreationDate = pd.to_datetime(df_answers.CreationDate).apply(lambda x: int(x.timestamp())) 
    df_answers = df_answers[df_answers['Score'] >= 0] # remove negative samples from data

    queries_file = f'{split}/data.jsonl'
    filename = os.path.join(data_folder, queries_file)
    test_data = load_test_query(filename)

    bm25_file = f'{split}/bm25_run.json'
    bm25_filename = os.path.join(data_folder, bm25_file)
    with open(bm25_filename, 'r') as f:
        test_bm25_run = json.load(f)

    test_question_tags, test_answer_tags = create_user_tags(test_data, df_question, df_answers, test_bm25_run, cpus)

    test_qrel, test_run = get_tag_rank(test_data, test_question_tags, test_answer_tags, test_bm25_run)
    
    test_qrel = Qrels(test_qrel)
    ranx_test_run = Run(test_run, name='Tag')
    ranx_bm25_run = Run(test_bm25_run, name='BM25')

    ranx_test_run.save(os.path.join(data_folder, f'{split}/tag_run.json'))

    models = [ranx_test_run, ranx_bm25_run]
    
    report = compare(
        qrels=test_qrel,
        runs=models,
        metrics=['mrr@10', 'ndcg@010', 'recall@10', 'precision@10', 'map@100', 'recall@100'],
        max_p=0.01  # P-value threshold
    )
    logger.info(f'\nReRank BM25 with TAG on {split} \n{report}')


if __name__ == "__main__":
    main()
    