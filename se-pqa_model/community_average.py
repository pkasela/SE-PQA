import logging
from os.path import join
import json

from ranx import Qrels, Run, compare, fuse, optimize_fusion

from dataloader.utils import load_test_query

import tqdm

logging.basicConfig(filename=join('../logs', 'answer_fusion.log'),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO
                    )

logger = logging.getLogger(__name__)

communities = [
    "writers",
    "workplace",
    "woodworking",
    "vegetarianism",
    "travel",
    "sustainability",
    "sports",
    "sound",
    "skeptics",
    "scifi",
    "rpg",
    "politics",
    "philosophy",
    "pets",
    "parenting",
    "outdoors",
    "opensource",
    "musicfans",
    "music",
    "movies",
    "money",
    "martialarts",
    "literature",
    "linguistics",
    "lifehacks",
    "law",
    "judaism",
    "islam",
    "interpersonal",
    "hsm",
    "history",
    "hinduism",
    "hermeneutics",
    "health",
    "genealogy",
    "gardening",
    "gaming",
    "freelancing",
    "fitness",
    "expatriates",
    "english",
    "diy",
    "cooking",
    "christianity",
    "buddhism",
    "boardgames",
    "bicycles",
    "apple",
    "anime",
    "academia"
]

communities_weights_2 = {
    'academia': {'weights': (0.1, 0.9)},
    'english': {'weights': (0.2, 0.8)},
    'gaming': {'weights': (0.1, 0.9)},
    'health': {'weights': (0.2, 0.8)},
    'cooking': {'weights': (0.1, 0.9)},
    'diy': {'weights': (0.1, 0.9)},
    'history': {'weights': (0.2, 0.8)},
    'law': {'weights': (0.2, 0.8)},
    'scifi': {'weights': (0.2, 0.8)},
    'travel': {'weights': (0.1, 0.9)},
    'workplace': {'weights': (0.3, 0.7)},
    'writers': {'weights': (0.2, 0.8)},
    'woodworking': {'weights': (0.2, 0.8)},
    'vegetarianism': {'weights': (0.3, 0.7)},
    'sustainability': {'weights': (0.1, 0.9)},
    'sports': {'weights': (0.2, 0.8)},
    'sound': {'weights': (0.2, 0.8)},
    'skeptics': {'weights': (0.2, 0.8)},
    'rpg': {'weights': (0.2, 0.8)},
    'politics': {'weights': (0.1, 0.9)},
    'philosophy': {'weights': (0.2, 0.8)},
    'pets': {'weights': (0.1, 0.9)},
    'parenting': {'weights': (0.1, 0.9)},
    'outdoors': {'weights': (0.1, 0.9)},
    'opensource': {'weights': (0.2, 0.8)},
    'musicfans': {'weights': (0.1, 0.9)},
    'music': {'weights': (0.2, 0.8)},
    'movies': {'weights': (0.1, 0.9)},
    'money': {'weights': (0.2, 0.8)},
    'martialarts': {'weights': (0.1, 0.9)},
    'literature': {'weights': (0.3, 0.7)},
    'linguistics': {'weights': (0.2, 0.8)},
    'lifehacks': {'weights': (0.1, 0.9)},
    'judaism': {'weights': (0.2, 0.8)},
    'islam': {'weights': (0.1, 0.9)},
    'interpersonal': {'weights': (0.2, 0.8)},
    'hsm': {'weights': (0.2, 0.8)},
    'hinduism': {'weights': (0.2, 0.8)},
    'hermeneutics': {'weights': (0.2, 0.8)},
    'genealogy': {'weights': (0.3, 0.7)},
    'gardening': {'weights': (0.1, 0.9)},
    'freelancing': {'weights': (0.1, 0.9)},
    'fitness': {'weights': (0.2, 0.8)},
    'expatriates': {'weights': (0.1, 0.9)},
    'christianity': {'weights': (0.2, 0.8)},
    'buddhism': {'weights': (0.3, 0.7)},
    'boardgames': {'weights': (0.1, 0.9)},
    'bicycles': {'weights': (0.1, 0.8, 0.1)},
    'apple': {'weights': (0.1, 0.9)},
    'anime': {'weights': (0.1, 0.9)},
}

communities_weights_3 = {
    'academia': {'weights': (0.1, 0.8, 0.1)},
    'english': {'weights': (0.2, 0.8, 0.0)},
    'gaming': {'weights': (0.1, 0.8, 0.1)},
    'health': {'weights': (0.2, 0.8, 0.0)},
    'cooking': {'weights': (0.1, 0.8, 0.1)},
    'diy': {'weights': (0.1, 0.8, 0.1)},
    'history': {'weights': (0.2, 0.8, 0.0)},
    'law': {'weights': (0.1, 0.8, 0.1)},
    'scifi': {'weights': (0.1, 0.8, 0.1)},
    'travel': {'weights': (0.1, 0.9, 0.0)},
    'workplace': {'weights': (0.3, 0.7, 0.0)},
    'writers': {'weights': (0.2, 0.8, 0.0)},
    'woodworking': {'weights': (0.2, 0.8, 0)},
    'vegetarianism': {'weights': (0.3, 0.7, 0)},
    'sustainability': {'weights': (0.1, 0.8, 0.1)},
    'sports': {'weights': (0.1, 0.8, 0.1)},
    'sound': {'weights': (0.1, 0.8, 0.1)},
    'skeptics': {'weights': (0.2, 0.8, 0)},
    'rpg': {'weights': (0.1, 0.8, 0.1)},
    'politics': {'weights': (0.1, 0.9, 0.1)},
    'philosophy': {'weights': (0.2, 0.8, 0)},
    'pets': {'weights': (0.1, 0.8, 0.1)},
    'parenting': {'weights': (0.1, 0.9, 0)},
    'outdoors': {'weights': (0.1, 0.9, 0)},
    'opensource': {'weights': (0.1, 0.8, 0.1)},
    'musicfans': {'weights': (0.1, 0.9, 0)},
    'music': {'weights': (0.1, 0.8, 0.1)},
    'movies': {'weights': (0.1, 0.8, 0.1)},
    'money': {'weights': (0.1, 0.8, 0.1)},
    'martialarts': {'weights': (0.1, 0.8, 0.1)},
    'literature': {'weights': (0.3, 0.7, 0)},
    'linguistics': {'weights': (0.2, 0.8, 0)},
    'lifehacks': {'weights': (0.1, 0.8, 0.1)},
    'judaism': {'weights': (0.2, 0.8, 0)},
    'islam': {'weights': (0.1, 0.8, 0.1)},
    'interpersonal': {'weights': (0.2, 0.8, 0)},
    'hsm': {'weights': (0.2, 0.8, 0)},
    'hinduism': {'weights': (0.1, 0.8, 0.1)},
    'hermeneutics': {'weights': (0.1, 0.8, 0.1)},
    'genealogy': {'weights': (0.3, 0.7, 0.3)},
    'gardening': {'weights': (0.1, 0.8, 0.1)},
    'freelancing': {'weights': (0.1, 0.9, 0)},
    'fitness': {'weights': (0.2, 0.8, 0)},
    'expatriates': {'weights': (0.1, 0.9, 0)},
    'christianity': {'weights': (0.1, 0.8, 0.1)},
    'buddhism': {'weights': (0.3, 0.7, 0)},
    'boardgames': {'weights': (0.1, 0.8, 0.1)},
    'bicycles': {'weights': (0.1, 0.8, 0.1)},
    'apple': {'weights': (0.1, 0.8, 0.1)},
    'anime': {'weights': (0.1, 0.9, 0)},
}

all_qrels = {}
all_bm25_bert_runs = {}
all_bm25_bert_tags_runs = {}
for comm in tqdm.tqdm(communities):

    data_folder = f'../dataset_{comm}/answer_retrieval'
    print(comm)
    test_queries = load_test_query(join(data_folder, 'test/data.jsonl'))
    for q in test_queries:
        all_qrels[q] = {str(k): 1 if test_queries[q].get('best_answer', 'no best answer') != str(k) else 2 for k in test_queries[q]['relevant_docs']} 

    split = 'test'
    model_epoch = 0

    with open(join(data_folder, f'{split}/bert_run_{model_epoch}_rerank.json'), 'r') as f:
        bert_run = json.load(f)
    bert_run = Run(bert_run, name='BERT')


    with open(join(data_folder, f'{split}/tag_run.json'), 'r') as f:
        tag_run = json.load(f)
    tag_run = Run(tag_run, name='TAG')

    with open(join(data_folder, f'{split}/bm25_run.json'), 'r') as f:
        bm25_run = json.load(f)
    bm25_run = Run(bm25_run, name='BM25')

    all_combined_test_run = fuse(
            runs=[bm25_run, bert_run, tag_run],  
            norm="min-max",       
            method="wsum",        
            params=communities_weights_3[comm],
        )
    all_combined_test_run.name = 'BM25 + BERT + TAG'


    bm25_bert_combined_test_run = fuse(
        runs=[bm25_run, bert_run],  
        norm="min-max",       
        method="wsum",        
        params=communities_weights_2[comm],
    )
    bm25_bert_combined_test_run.name = 'BM25 + BERT'

    bm25_bert_combined_test_run_dict = bm25_bert_combined_test_run.to_dict()

    all_combined_test_run_dict = all_combined_test_run.to_dict()

    for r in bm25_bert_combined_test_run_dict:
        all_bm25_bert_runs[r] = bm25_bert_combined_test_run_dict[r]

    for r in all_combined_test_run_dict:
        all_bm25_bert_tags_runs[r] = all_combined_test_run_dict[r]


all_bm25_bert_runs_ranx = Run(all_bm25_bert_runs, name='BM25+BERT')
all_bm25_bert_tags_runs_ranx = Run(all_bm25_bert_tags_runs, name='BM25+BERT+TAG')
all_qrels_ranx = Qrels(all_qrels)

models = [
    all_bm25_bert_runs_ranx, 
    all_bm25_bert_tags_runs_ranx
]

logger.info('Comparing test results')
report = compare(
    qrels=all_qrels_ranx,
    runs=models,
    metrics=['precision@1', 'ndcg@3', 'ndcg@10', 'recall@100', 'map@100'],#['map@100', 'mrr@100', 'ndcg@100', 'mrr@10', 'ndcg@010'],
    max_p=0.01  # P-value threshold
)
logger.info(f'\n{report}')
print(report)

    