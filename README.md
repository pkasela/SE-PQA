# SE-PQA: a Resource for Personalized Community Question Answering

## To create the datset

The dataset is provided already processed on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10679181.svg)](https://doi.org/10.5281/zenodo.10679181)

To recreate the dataset: 
1) Run the `download_and_extract_data.sh` file to download the dataset from archieve (the version changes overtime, will upload the one used by me somewhere)
2) Run the `combine_data.py` file to combine all the various community into one
3) go to the `se-pqa_data` folder and run the `pipeline.sh` command to create the files as uploaded on zenodo.

## To create the baseline:

To create the baseline provided in the paper run the `pipeline.sh` file in `se-pqa_model` folder (change the flag values to adjust file paths).


## Abstract 

Personalization in Information Retrieval is a topic studied for a long time. Nevertheless, there is still a lack of high-quality, real-world datasets to conduct large-scale experiments and evaluate models for personalized search. This paper contributes to fill this gap by introducing SE-PQA (StackExchange - Personalized Question Answering), a new resource to design and evaluate personalized models related to the two tasks of community Question Answering (cQA). The contributed dataset includes more than  1 million queries and 2 million answers,  annotated with a rich set of features modeling the social interactions among the users of a popular cQA platform. We describe the characteristics of SE-PQA and detail the features associated with both questions and answers. We also provide reproducible baseline methods for the cQA task based on the resource, including deep learning models and personalization approaches. The results of the preliminary experiments conducted show the appropriateness of SE-PQA to train effective cQA models; they also show that personalization improves remarkably the effectiveness of all the methods tested. Furthermore, we show the benefits in terms of robustness and generalization of combining data from multiple communities for personalization purposes.


```
@misc{kasela2024sepqa,
      title={SE-PQA: Personalized Community Question Answering}, 
      author={Pranav Kasela and Marco Braga and Gabriella Pasi and Raffaele Perego},
      year={2024},
      eprint={2306.16261},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```
