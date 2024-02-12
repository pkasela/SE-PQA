import json
import logging
import os
from os import makedirs
from os.path import join

import click
import torch
from adapters import BnConfig, init
from dataloader.dataloader import StarQuestionData, in_batch_negative_collate_fn_bm25_t5
from dataloader.utils import load_query_data, seed_everything
from torch import cuda
from torch.utils.data import DataLoader
from transformers import AdamW, T5ForConditionalGeneration

os.environ['WANDB_DISABLED'] = 'true'

@click.command()
@click.option(
    "--data_folder",
    type=str,
    required=True,
)
@click.option(
    "--num_epochs",
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
    "--model_name",
    type=str,
    required=True
)
@click.option(
    "--lr",
    type=float,
    required=True
)
@click.option(
    "--output_dir",
    type=str,
    required=True
)
@click.option(
    "--reduction_factor",
    type=int,
    default=None
)
def main(data_folder, num_epochs, batch_size, seed, model_name, lr, output_dir, reduction_factor):
    makedirs('../logs', exist_ok=True)
    logging_file = f"training_t5_{data_folder.split('/')[-1]}.log"
    logging.basicConfig(
        filename=join('../logs', logging_file),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    device = 'cuda' if cuda.is_available() else 'cpu'

    seed_everything(seed)

    with open(join(data_folder, 'answer_collection.json'), 'r') as f:
        corpus = json.load(f)

    train_query_data = load_query_data(join(data_folder, 'train/queries.jsonl'))
    val_query_data = load_query_data(join(data_folder, 'val/queries.jsonl'))

    train_data = DataLoader(
        StarQuestionData(train_query_data, corpus),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=in_batch_negative_collate_fn_bm25_t5
    ) 

    eval_data = DataLoader(
        StarQuestionData(val_query_data, corpus),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=in_batch_negative_collate_fn_bm25_t5
    )

    model = T5ForConditionalGeneration.from_pretrained(model_name)
    # Enable adapter support
    init(model)
    config = BnConfig(mh_adapter=False, output_adapter=True, reduction_factor=reduction_factor, non_linearity="relu")
    # Add new adapter
    model.add_adapter("t5_qa", config = config)

    # Activate adapter for training
    model.train_adapter("t5_qa")

    for _, p in model.named_parameters():
        if p.requires_grad is True:
            print(p)

    model = model.to(device)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad is True], lr=lr)

    loss_history = []
    for epoch in range(num_epochs):
        train_loss = 0
        model.train(True)
        for b, batch in enumerate(train_data):
            optimizer.zero_grad()
            outputs = model(input_ids = batch['input_ids'].to(device), labels = batch['labels'].to(device))
            loss = outputs.loss
            loss.backward()
            loss_history.append(loss.item())
            train_loss += loss.item()
            batch_loss = loss.item()
            optimizer.step()
            if b%1000 == 0:
                logging.info("TRAIN batch {:3d} Current loss {:.2e}".format(b, batch_loss))
            
        
        print("TRAIN EPOCH {:3d} Current loss {:.2e}".format(epoch, train_loss))
        # Evaluation
        model.eval()
        loss_eval = 0
        with torch.no_grad():
            for batch in eval_data:
                outputs = model(input_ids = batch['input_ids'].to('cuda'), labels = batch['labels'].to('cuda'))

                loss_eval += outputs.loss.item()
                

        logging.info("VAL EPOCH {:3d} Current loss {:.2e}".format(epoch, loss_eval))

        output_dir_ep = output_dir +'_epoch_'+str(epoch)
        if not os.path.exists(output_dir_ep):
            os.makedirs(output_dir_ep)
        
        model.save_adapter(output_dir_ep, "t5_qa")



    output_dir = output_dir +'_end_training'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_adapter(output_dir, "t5_qa")


if __name__ == '__main__':
    main()