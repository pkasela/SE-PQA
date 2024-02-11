DATA_FOLDER='../dataset/answer_retrieval'
BERT='distilbert-base-uncased' #'nreimers/MiniLM-L6-H384-uncased'
EPOCH=10
BATCH=16
LR=1e-6
MODE='bm25'
SEED=42
MODEL_OUTPUT='./saved_models'

DF_FOLDER='../dataset'
CPUS=10
SPLIT='val'
python testing_pers_tag.py --df_folder $DF_FOLDER --data_folder $DATA_FOLDER --split $SPLIT --seed $SEED --cpus $CPUS
SPLIT='test'
python testing_pers_tag.py --df_folder $DF_FOLDER --data_folder $DATA_FOLDER --split $SPLIT --seed $SEED --cpus $CPUS


python training_dense.py --data_folder $DATA_FOLDER --max_epoch $EPOCH --batch_size $BATCH --seed $SEED --mode $MODE --bert_name $BERT --lr $LR --output_folder $MODEL_OUTPUT

SAVED_MODEL=$MODEL_OUTPUT'/model_10.pt'
OUTPUT_FOLDER='../created_data/03_best_answer'
EMB_DIM=768
BATCH=64

python create_answer_embeddings.py --data_folder $DATA_FOLDER --embedding_dim $EMB_DIM --bert_name $BERT --batch_size $BATCH --seed $SEED --saved_model $SAVED_MODEL --output_folder $OUTPUT_FOLDER

SPLIT='val'
python testing_dense.py --data_folder $DATA_FOLDER --output_folder $OUTPUT_FOLDER --bert_name $BERT --model_path $SAVED_MODEL --split $SPLIT --seed $SEED
SPLIT='test'
python testing_dense.py --data_folder $DATA_FOLDER --output_folder $OUTPUT_FOLDER --bert_name $BERT --model_path $SAVED_MODEL --split $SPLIT --seed $SEED

python fuse_optimizer.py --data_folder $DATA_FOLDER --model_path $SAVED_MODEL --mode 'base'
python fuse_optimizer.py --data_folder $DATA_FOLDER --model_path $SAVED_MODEL --mode 'pers'

SAVED_MODEL=$MODEL_OUTPUT'/model_0.pt'
BERT='nreimers/MiniLM-L6-H384-uncased'
EMB_DIM=384
BATCH=64
python create_model_zero.py --bert_name $BERT --saved_model $SAVED_MODEL
python create_answer_embeddings.py --data_folder $DATA_FOLDER --embedding_dim $EMB_DIM --bert_name $BERT --batch_size $BATCH --seed $SEED --saved_model $SAVED_MODEL --output_folder $OUTPUT_FOLDER

SPLIT='val'
python testing_dense.py --data_folder $DATA_FOLDER --output_folder $OUTPUT_FOLDER --bert_name $BERT --model_path $SAVED_MODEL --split $SPLIT --seed $SEED
SPLIT='test'
python testing_dense.py --data_folder $DATA_FOLDER --output_folder $OUTPUT_FOLDER --bert_name $BERT --model_path $SAVED_MODEL --split $SPLIT --seed $SEED

python fuse_optimizer.py --data_folder $DATA_FOLDER --model_path $SAVED_MODEL --mode 'base'
python fuse_optimizer.py --data_folder $DATA_FOLDER --model_path $SAVED_MODEL --mode 'pers'



SEED=0
MODEL_NAME="castorini/monot5-small-msmarco-10k"
LR=1e-3
OUTPUT_DIR='./t5_sepqa_new_len_model_small'
REDUCTION_FACTOR=48
BATCH=64

python training_t5_adapter.py --data_folder $DATA_FOLDER --num_epochs $EPOCH --batch_size $BATCH --seed $SEED --model_name $MODEL_NAME --lr $LR --output_dir $OUTPUT_DIR --reduction_factor $REDUCTION_FACTOR

SEED=42
SPLIT='val'
python testing_t5_adapter.py --data_folder $DATA_FOLDER --model_path $OUTPUT_DIR --split $SPLIT --seed $SEED
SPLIT='test'
python testing_t5_adapter.py --data_folder $DATA_FOLDER --model_path $OUTPUT_DIR --split $SPLIT --seed $SEED

python fuse_optimizer_t5.py --data_folder $DATA_FOLDER --model_name $OUTPUT_DIR --mode 'base'
python fuse_optimizer_t5.py --data_folder $DATA_FOLDER --model_name $OUTPUT_DIR --mode 'pers'

MODEL_NAME="castorini/monot5-base-msmarco-10k"
OUTPUT_DIR='./t5_sepqa_new_len_model_base'

python training_t5_adapter.py --data_folder $DATA_FOLDER --num_epochs $EPOCH --batch_size $BATCH --seed $SEED --model_name $MODEL_NAME --lr $LR --output_dir $OUTPUT_DIR --reduction_factor $REDUCTION_FACTOR

SEED=42
SPLIT='val'
python testing_t5_adapter.py --data_folder $DATA_FOLDER --model_path $OUTPUT_DIR --split $SPLIT --seed $SEED
SPLIT='test'
python testing_t5_adapter.py --data_folder $DATA_FOLDER --model_path $OUTPUT_DIR --split $SPLIT --seed $SEED

python fuse_optimizer_t5.py --data_folder $DATA_FOLDER --model_name $OUTPUT_DIR --mode 'base'
python fuse_optimizer_t5.py --data_folder $DATA_FOLDER --model_name $OUTPUT_DIR --mode 'pers'
