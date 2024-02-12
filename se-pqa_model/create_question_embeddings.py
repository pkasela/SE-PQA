import json
import logging
import os

import click
import torch
import tqdm
from dataloader.utils import seed_everything
from model.model import BiEncoder
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

@click.command()
@click.option(
    "--data_folder",
    type=str,
    required=True,
)
@click.option(
    "--embedding_dim",
    type=int,
    required=True
)
@click.option(
    "--bert_name",
    type=str,
    required=True
)
@click.option(
    "--batch_size",
    type=int,
    required=True,
)
@click.option(
    "--seed",
    type=int,
    default=None
)
@click.option(
    "--saved_model",
    type=str,
    default=None
)
@click.option(
    "--output_folder",
    type=str,
    required=True
)
def main(data_folder, embedding_dim, bert_name, batch_size, seed, saved_model, output_folder):
    if seed:
        seed_everything(seed)
    collection_file = os.path.join(data_folder, 'question_collection.json')
    with open(collection_file, 'r') as f:
        corpus = json.load(f)
    
    embedding_matrix = torch.zeros(len(corpus), embedding_dim).float()
    logging.info(f'Embedding Matrix dimentions: {embedding_matrix.shape}')

    logging.debug('Loading model and tokenizer')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    doc_model = AutoModel.from_pretrained(bert_name)
    model = BiEncoder(doc_model, tokenizer, device)

    model_epoch = int(saved_model.replace('.pt', '').split('_')[-1]) if saved_model else 0
    if saved_model:
        logging.info(f'Loading model at path {saved_model}')
        model.load_state_dict(torch.load(saved_model))

    index = 0
    batch_val = 0
    texts = []
    id_to_index = {}
    for id_, val in tqdm.tqdm(corpus.items()):
        id_to_index[id_] = index
        batch_val += 1
        index += 1
        if type(val) != 'str':
            val = str(val)
        texts.append(val)
        if batch_val == batch_size:
            with torch.no_grad():
                embedding_matrix[index - batch_val : index] = model.doc_encoder(texts).cpu()
            batch_val = 0
            texts = []

    if texts:
        embedding_matrix[index - batch_val : index, :] = model.doc_encoder(texts).cpu()

    os.makedirs(output_folder, exist_ok=True)
    logging.info(f'Embedded {index} documents. Saving embedding matrix in folder {output_folder}.')
    if saved_model:
        torch.save(embedding_matrix, os.path.join(output_folder, f'question_embedding_{model_epoch}.pt'))
    else:
        torch.save(embedding_matrix, os.path.join(output_folder, 'question_embedding_base.pt'))

    logging.info(f'Embedded {index} documents. Saving id_to_index.json in folder {output_folder}.')
    if saved_model:
        with open(os.path.join(output_folder, f'qid_to_index_{model_epoch}.json'), 'w') as f:
            json.dump(id_to_index, f)
    else:
        with open(os.path.join(output_folder, 'qid_to_index_base.json'), 'w') as f:
            json.dump(id_to_index, f)
            
if __name__ == '__main__':
    main()
