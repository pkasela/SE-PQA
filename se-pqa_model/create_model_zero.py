import click
import torch
from model.model import BiEncoder
from transformers import AutoModel, AutoTokenizer


@click.command()
@click.option(
    "--bert_name",
    type=str,
    required=True,
)
@click.option(
    "--saved_model",
    type=str,
    required=True,
)
def main(bert_name, saved_model):
    doc_model = AutoModel.from_pretrained(bert_name)
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BiEncoder(doc_model, tokenizer, device)

    torch.save(model.state_dict(), saved_model)

if __name__ == '__main__':
    main()