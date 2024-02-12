import json
import logging
from os import makedirs
from os.path import join

import click
import numpy as np
from dataloader.dataloader import (
    QuestionData,
    StarQuestionData,
    in_batch_negative_collate_fn,
)
from dataloader.utils import load_query_data, load_test_query, seed_everything
from model.loss import TripletMarginLoss
from model.model import BiEncoder
from torch import cuda, load, no_grad, save
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

def train(train_query_data, corpus, batch_size, optimizer, loss_fn, model, epoch, mode):
    alpha = 0.2
    if mode == 'base':
        train_data = tqdm(DataLoader(QuestionData(train_query_data, corpus), batch_size=batch_size, shuffle=True, collate_fn=in_batch_negative_collate_fn))
    else:
        train_data = tqdm(DataLoader(StarQuestionData(train_query_data, corpus), batch_size=batch_size, shuffle=True, collate_fn=in_batch_negative_collate_fn))
    losses = []
    optimizer.zero_grad()
    for step, triples in enumerate(train_data):
        triple = triples['question'], triples['pos_text'], triples['neg_text']
        with cuda.amp.autocast():
            query_embedding, pos_embedding, neg_embedding = model.forward_random_neg(triple)
            loss_val = loss_fn(query_embedding, pos_embedding, neg_embedding)

        if mode == 'bm25':
            triple = triples['question'], triples['pos_text'], triples['bm25_neg_text']
            with cuda.amp.autocast():
                query_embedding, pos_embedding, neg_embedding = model(triple)
                loss_val += alpha*loss_fn(query_embedding, pos_embedding, neg_embedding)
            
        loss_val.backward()
            
        # if (step % 4) == 0:
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss_val.cpu().detach().item())
        average_loss = np.mean(losses)
        train_data.set_description("TRAIN EPOCH {:3d} Current loss {:.2e}, Average {:.2e}".format(epoch, loss_val, average_loss))
    return average_loss

def validate(val_query_data, corpus, batch_size, loss_fn, model, epoch, mode):
    if mode == 'base':
        val_data = tqdm(DataLoader(QuestionData(val_query_data, corpus), batch_size=batch_size, shuffle=False, collate_fn=in_batch_negative_collate_fn))
    else:
        val_data = tqdm(DataLoader(StarQuestionData(val_query_data, corpus), batch_size=batch_size, shuffle=False, collate_fn=in_batch_negative_collate_fn))
    losses = []
    for triple in val_data:
        triple = triple['question'], triple['pos_text'], triple['neg_text']
        with no_grad():
            query_embedding, pos_embedding, neg_embedding = model.forward_random_neg(triple)
            loss_val = loss_fn(query_embedding, pos_embedding, neg_embedding)
            losses.append(loss_val.cpu().detach().item())
            
        average_loss = np.mean(losses)
        val_data.set_description("Val EPOCH {:3d} Current loss {:.2e}, Average {:.2e}".format(epoch, loss_val, average_loss))

    return average_loss

@click.command()
@click.option(
    "--data_folder",
    type=str,
    required=True,
)
@click.option(
    "--max_epoch",
    type=int,
    required=True,
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
    "--mode",
    type=str,
    required=True
)
@click.option(
    "--bert_name",
    type=str,
    required=True
)
@click.option(
    "--lr",
    type=float,
    required=True
)
@click.option(
    "--output_folder",
    type=str,
    required=True
)
@click.option(
    "--saved_model",
    type=str,
    default=None
)
def main(data_folder, max_epoch, batch_size, seed, mode, bert_name, lr, output_folder, saved_model):
    makedirs('../logs', exist_ok=True)
    logging_file = f"training_{data_folder.split('/')[-1]}.log"
    logging.basicConfig(filename=join('../logs', logging_file),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO
                        )

    device = 'cuda' if cuda.is_available() else 'cpu'
    
    assert mode in {'bm25', 'base'}, 'mode is allowed to be either bm25 or base'
    if seed:
        seed_everything(seed)
    makedirs(output_folder, exist_ok=True)

    with open(join(data_folder, 'answer_collection.json'), 'r') as f:
        corpus = json.load(f)

    if mode == 'base':
        train_query_data = load_test_query(join(data_folder, 'train/data.jsonl'))
        val_query_data = load_test_query(join(data_folder, 'val/data.jsonl'))
    else:
        train_query_data = load_query_data(join(data_folder, 'train/queries.jsonl'))
        val_query_data = load_query_data(join(data_folder, 'val/queries.jsonl'))

    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    doc_model = AutoModel.from_pretrained(bert_name)
    loss_fn = TripletMarginLoss(.5).to(device)
    model = BiEncoder(doc_model, tokenizer, device)
    start = 0
    if saved_model:
        logging.info(f'Loading model in {saved_model}')
        model.load_state_dict(load(saved_model))
        start = int(saved_model.replace('.pt', '').split('_')[-1])
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(start + 1, start + max_epoch + 1):
        average_loss = train(train_query_data, corpus, batch_size, optimizer, loss_fn, model, epoch, mode)

        logging.info("TRAIN EPOCH: {:3d}, Average Loss: {:.2e}".format(epoch, average_loss))

        model_name = join(output_folder, f"model_{epoch}.pt")
        save(model.state_dict(), model_name)

        average_loss = validate(val_query_data, corpus, batch_size, loss_fn, model, epoch, mode)  
        
        logging.info("VAL EPOCH: {:3d}, Average Loss: {:.2e}".format(epoch, average_loss))
        

if __name__ == '__main__':
    main()