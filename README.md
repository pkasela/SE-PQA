# SE-PQA

## To create the datset

The dataset is provided already processed on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10679181.svg)](https://doi.org/10.5281/zenodo.10679181)

To recreate the dataset: 
1) Run the `download_and_extract_data.sh` file to download the dataset from archieve (the version changes overtime, will upload the one used by me somewhere)
2) Run the `combine_data.py` file to combine all the various community into one
3) go to the `se-pqa_data` folder and run the `pipeline.sh` command to create the files as uploaded on zenodo.

## To create the baseline:

To create the baseline provided in the paper run the `pipeline.sh` file in `se-pqa_model` folder (change the flag values to adjust file paths).


