import json
import logging
from os.path import join

import click
from dataloader.utils import load_test_query
from ranx import Qrels, Run, compare, fuse, optimize_fusion

logging.basicConfig(filename=join('../logs', 'answer_fusion.log'),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO
                    )

logger = logging.getLogger(__name__)

def test_t5_tag(data_folder, model_name, mode):
    assert mode in ['base', 'pers'], "model can only be base or pers"
    
    logger.info(f'Model name: {model_name}, mode: {mode}')
    logger.info('Creating Validation and Test Qrels')

    if mode == "base":
        val_queries = load_test_query(join(data_folder, 'val/data.jsonl'))
        val_qrels = {q: {str(k): 1 if val_queries[q].get('best_answer', 'no best answer') != str(k) else 2 for k in val_queries[q]['relevant_docs']} for q in val_queries}

        test_queries = load_test_query(join(data_folder, 'test/data.jsonl'))
        test_qrels = {q: {str(k): 1 if test_queries[q].get('best_answer', 'no best answer') != str(k) else 2 for k in test_queries[q]['relevant_docs']} for q in test_queries}
    else:
        val_queries = load_test_query(join(data_folder, 'val/data_pers.jsonl'))
        val_qrels = {q: {str(k): 1 for k in val_queries[q]['relevant_docs']} for q in val_queries}

        test_queries = load_test_query(join(data_folder, 'test/data_pers.jsonl'))
        test_qrels = {q: {str(k): 1 for k in test_queries[q]['relevant_docs']} for q in test_queries}
    
    
    split = 'val'
    
    logger.info('Reading Validation Runs')
    with open(join(data_folder, f'{split}/{model_name.replace("/","_")}_rerank.json'), 'r') as f:
        t5_run = json.load(f)
    if mode == 'pers':
        t5_run = {q: t5_run[q] for q in t5_run if q in val_qrels}
    t5_run = Run(t5_run, name='T5')

    with open(join(data_folder, f'{split}/tag_set_run.json'), 'r') as f:
        tag_run = json.load(f)
    if mode == 'pers':
        tag_run = {q: tag_run[q] for q in tag_run if q in val_qrels}
    tag_run = Run(tag_run, name='TAG')

    with open(join(data_folder, f'{split}/bm25_run.json'), 'r') as f:
        bm25_run = json.load(f)
    if mode == 'pers':
        bm25_run = {q: bm25_run[q] for q in bm25_run if q in val_qrels}
    bm25_run = Run(bm25_run, name='BM25')

    val_qrels = Qrels(val_qrels)  

    logger.info('Optimizing on validation')
    all_best_params = optimize_fusion(
        qrels=val_qrels,
        runs=[bm25_run, t5_run, tag_run],
        norm="min-max",
        method="wsum",
        metric="ndcg@10",  # The metric to maximize during optimization
        return_optimization_report=True
    )

    bm25_bert_best_params = optimize_fusion(
        qrels=val_qrels,
        runs=[bm25_run, t5_run],
        norm="min-max",
        method="wsum",
        metric="ndcg@10",  # The metric to maximize during optimization
        return_optimization_report=True
    )

    bm25_tag_best_params = optimize_fusion(
        qrels=val_qrels,
        runs=[bm25_run, tag_run],
        norm="min-max",
        method="wsum",
        metric="ndcg@10",  # The metric to maximize during optimization
        return_optimization_report=True
    )

    logger.info(f'best parameters with all three: {all_best_params[0]}')
    logger.info(f'\n{all_best_params[1]}')


    logger.info(f'best parameters with bm25 and bert: {bm25_bert_best_params[0]}')
    logger.info(f'\n{bm25_bert_best_params[1]}')

    logger.info(f'best parameters with bm25 and tag: {bm25_tag_best_params[0]}')
    logger.info(f'\n{bm25_tag_best_params[1]}')
    
    # all_best_params = ({'weights': (0., 1, 0)}, [])
    # bm25_bert_best_params = ({'weights': (0, 1)}, [])
    # bm25_tag_best_params = ({'weights': (0.7, 0.3)}, [])
    logger.info('Reading test run.')
    split = 'test'

    with open(join(data_folder, f'{split}/{model_name.replace("/","_")}_rerank.json'), 'r') as f:
        t5_run = json.load(f)
    if mode == 'pers':
        t5_run = {q: t5_run[q] for q in t5_run if q in test_qrels}
    t5_run = Run(t5_run, name='T5')
    logger.info('Read T5 run')


    with open(join(data_folder, f'{split}/tag_set_run.json'), 'r') as f:  #tag_embedding_run, tag_run tag_embedding_den_run
        tag_run = json.load(f)
    if mode == 'pers':
        tag_run = {q: tag_run[q] for q in tag_run if q in test_qrels}
    tag_run = Run(tag_run, name='TAG')
    logger.info('Read TAG Run')


    with open(join(data_folder, f'{split}/bm25_run.json'), 'r') as f:
        bm25_run = json.load(f)
    if mode == 'pers':
        bm25_run = {q: bm25_run[q] for q in bm25_run if q in test_qrels}
    bm25_run = Run(bm25_run, name='BM25')
    logger.info('Read BM25 Run')

    logger.info('Creating Test Qrels')
    test_qrels = Qrels(test_qrels)
    logger.info('Test Qrels Created')

    all_combined_test_run = fuse(
        runs=[bm25_run, t5_run, tag_run],  
        norm="min-max",       
        method="wsum",        
        params=all_best_params[0],
    )
    all_combined_test_run.name = 'BM25 + T5 + TAG'
    
    bm25_bert_combined_test_run = fuse(
        runs=[bm25_run, t5_run],  
        norm="min-max",       
        method="wsum",        
        params=bm25_bert_best_params[0],
    )
    bm25_bert_combined_test_run.name = 'BM25 + T5'
    
    bm25_tag_combined_test_run = fuse(
        runs=[bm25_run, tag_run],  
        norm="min-max",       
        method="wsum",        
        params=bm25_tag_best_params[0],
    )
    bm25_tag_combined_test_run.name = 'BM25 + TAG'

    logger.info('Test Fusion completed')
    


    models = [
        bm25_run,
        t5_run, 
        tag_run, 
        bm25_tag_combined_test_run, 
        bm25_bert_combined_test_run, 
        all_combined_test_run
    ]
    
    logger.info('Comparing test results')
    report = compare(
        qrels=test_qrels,
        runs=models,
        metrics=['precision@1', 'ndcg@3', 'ndcg@10', 'recall@100', 'map@100'],
        max_p=0.01/3  # P-value threshold, 3 tests
    )
    logger.info(f'\n{report}')
    print(report)

@click.command()
@click.option(
    "--data_folder",
    type=str,
    required=True
)
@click.option(
    "--model_name",
    type=str,
    required=True
)
@click.option(
    "--mode",
    type=str,
    required=True
)
def main(data_folder, model_name, mode):
    model_name = model_name.replace('/', '_')
    test_t5_tag(data_folder, model_name, mode)

if __name__ == "__main__":
    main()
