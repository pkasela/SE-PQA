import json
from os.path import join
import click
from tqdm import tqdm

def load_jsonl(file: str):
    with open(file, 'r') as f:
        for lne in f:
            yield json.loads(lne)

def combine_qrel_run(data_folder, split, top_k):
    with open(join(data_folder, f'{split}/bm25_run.json'), 'r') as f:
        bm25_runs = json.load(f)

    qrels = load_jsonl(join(data_folder, f'{split}/data.jsonl'))

    final_jsonl = []
    for q in tqdm(qrels, desc=f'Combining {split} Dictionary'):
        q['bm25_doc_ids'] = list(bm25_runs[q['id']].keys())[:top_k]
        q['bm25_doc_scores'] = list(bm25_runs[q['id']].values())[:top_k]
        
        final_jsonl.append(q)

    with open(join(data_folder, f'{split}/queries.jsonl'), 'w') as f:
        for row in tqdm(final_jsonl, desc=f'Writing {split} jsonl'):
            json.dump(row, f)
            f.write('\n')

    
    pers_qrels = load_jsonl(join(data_folder, f'{split}/data_pers.jsonl'))

    pers_final_jsonl = []
    for q in tqdm(pers_qrels, desc=f'Combining {split} Dictionary'):
        q['bm25_doc_ids'] = list(bm25_runs[q['id']].keys())[:top_k]
        q['bm25_doc_scores'] = list(bm25_runs[q['id']].values())[:top_k]
        
        pers_final_jsonl.append(q)

    with open(join(data_folder, f'{split}/queries_pers.jsonl'), 'w') as f:
        for row in tqdm(pers_final_jsonl, desc=f'Writing {split} jsonl'):
            json.dump(row, f)
            f.write('\n')
        
@click.command()
@click.option(
    "--dataset_folder",
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
def main(dataset_folder, train_top_k, val_top_k, test_top_k):
    # dataset_folder = '../dataset/answer_retrieval'
    split = 'train'
    combine_qrel_run(dataset_folder, split, train_top_k)

    split = 'val'
    combine_qrel_run(dataset_folder, split, val_top_k)

    split = 'test'
    combine_qrel_run(dataset_folder, split, test_top_k)

if __name__ == '__main__':
    main()