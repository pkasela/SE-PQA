DATA_FOLDER='../dataset/answer_retrieval'
BERT='distilbert-base-uncased' #'nreimers/MiniLM-L6-H384-uncased' 
MODE='bm25'
SEED=42
MODEL_OUTPUT='./saved_models'

DF_FOLDER='../dataset'
CPUS=10
SPLIT='val'
python testing_pers_tag.py --df_folder $DF_FOLDER --data_folder $DATA_FOLDER --split $SPLIT --seed $SEED --cpus $CPUS
SPLIT='test'
python testing_pers_tag.py --df_folder $DF_FOLDER --data_folder $DATA_FOLDER --split $SPLIT --seed $SEED --cpus $CPUS

SAVED_MODEL=$MODEL_OUTPUT'/model_10.pt'
OUTPUT_FOLDER='../created_data/03_best_answer'
EMB_DIM=768
BATCH=64

python create_answer_embeddings.py --data_folder $DATA_FOLDER --embedding_dim $EMB_DIM --bert_name $BERT --batch_size $BATCH --seed $SEED --saved_model $SAVED_MODEL --output_folder $OUTPUT_FOLDER

SPLIT='val'
python testing_bm25.py --data_folder $DATA_FOLDER --output_folder $OUTPUT_FOLDER --bert_name $BERT --model_path $SAVED_MODEL --split $SPLIT --seed $SEED
SPLIT='test'
python testing_bm25.py --data_folder $DATA_FOLDER --output_folder $OUTPUT_FOLDER --bert_name $BERT --model_path $SAVED_MODEL --split $SPLIT --seed $SEED

python fuse_optimizer.py --data_folder $DATA_FOLDER --model_path $SAVED_MODEL --mode 'base'
python fuse_optimizer.py --data_folder $DATA_FOLDER --model_path $SAVED_MODEL --mode 'pers'

SAVED_MODEL=$MODEL_OUTPUT'/model_0.pt'
BERT='nreimers/MiniLM-L6-H384-uncased'
EMB_DIM=384
BATCH=64
python create_answer_embeddings.py --data_folder $DATA_FOLDER --embedding_dim $EMB_DIM --bert_name $BERT --batch_size $BATCH --seed $SEED --saved_model $SAVED_MODEL --output_folder $OUTPUT_FOLDER

SPLIT='val'
python testing_bm25.py --data_folder $DATA_FOLDER --output_folder $OUTPUT_FOLDER --bert_name $BERT --model_path $SAVED_MODEL --split $SPLIT --seed $SEED
SPLIT='test'
python testing_bm25.py --data_folder $DATA_FOLDER --output_folder $OUTPUT_FOLDER --bert_name $BERT --model_path $SAVED_MODEL --split $SPLIT --seed $SEED

python fuse_optimizer.py --data_folder $DATA_FOLDER --model_path $SAVED_MODEL --mode 'base'
python fuse_optimizer.py --data_folder $DATA_FOLDER --model_path $SAVED_MODEL --mode 'pers'

