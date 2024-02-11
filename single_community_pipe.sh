declare -a UrlArray=(
    'writers'
    'workplace'
    'woodworking'
    'vegetarianism'
    'travel'
    'sustainability'
    'sports'
    'sound'
    'skeptics'
    'scifi'
    'rpg'
    'politics'
    'philosophy'
    'pets'
    'parenting'
    'outdoors'
    'opensource'
    'musicfans'
    'music'
    'movies'
    'money'
    'martialarts'
    'literature'
    'linguistics'
    'lifehacks'
    'law'
    'judaism'
    'islam'
    'interpersonal'
    'hsm'
    'history'
    'hinduism'
    'hermeneutics'
    'health'
    'genealogy'
    'gardening'
    'gaming'
    'freelancing'
    'fitness'
    'expatriates'
    'english'
    'diy'
    'cooking'
    'christianity'
    'buddhism'
    'boardgames'
    'bicycles'
    'apple'
    'anime'
    'academia'
)

for COMM in "${UrlArray[@]}"; do
    if [ -d dataset_$COMM ]; then
        echo "$COMM exists. Skipping computation"
    else
        python combine_data_single_comm.py --community $COMM

        cd se-pqa_data

        python create_best_answer_data.py --dataset_folder ../dataset_$COMM --train_split_time '2019-12-31 23:59:59' --test_split_time '2020-12-31 23:59:59'

        python get_bm25_runs.py --dataset_folder ../dataset_$COMM/answer_retrieval --cpus 10 --index_name stack_answers --ip localhost --port 9200 --mapping_path mapping.json --train_top_k 100 --val_top_k 100 --test_top_k 100

        # python get_remaining_bm25_runs.py --dataset_folder ../dataset/answer_retrieval --index_name stack_answers --ip localhost --port 9200 --mapping_path mapping.json --train_top_k 100 --val_top_k 100 --test_top_k 100
        python create_final_run.py --dataset_folder ../dataset_$COMM/answer_retrieval --train_top_k 100 --val_top_k 100 --test_top_k 100

        cd ../se-pqa_model

        DATA_FOLDER=../dataset_$COMM/answer_retrieval
        SEED=42
        MODEL_OUTPUT='./saved_models'

        DF_FOLDER=../dataset_$COMM
        CPUS=10
        SPLIT='val'
        python testing_pers_tag.py --df_folder $DF_FOLDER --data_folder $DATA_FOLDER --split $SPLIT --seed $SEED --cpus $CPUS
        SPLIT='test'
        python testing_pers_tag.py --df_folder $DF_FOLDER --data_folder $DATA_FOLDER --split $SPLIT --seed $SEED --cpus $CPUS

        SAVED_MODEL=$MODEL_OUTPUT'/model_0.pt'
        BERT='nreimers/MiniLM-L6-H384-uncased'
        EMB_DIM=384
        BATCH=64
        OUTPUT_FOLDER='../created_data/03_best_answer'

        python create_model_zero.py --bert_name $BERT --saved_model $SAVED_MODEL
        # run the following if you have not created the embeddings for the whole collection before, other wise keep like this
        # python create_answer_embeddings.py --data_folder $DATA_FOLDER --embedding_dim $EMB_DIM --bert_name $BERT --batch_size $BATCH --seed $SEED --saved_model $SAVED_MODEL --output_folder $OUTPUT_FOLDER

        SPLIT='val'
        python testing_dense.py --data_folder $DATA_FOLDER --output_folder $OUTPUT_FOLDER --bert_name $BERT --model_path $SAVED_MODEL --split $SPLIT --seed $SEED
        SPLIT='test'
        python testing_dense.py --data_folder $DATA_FOLDER --output_folder $OUTPUT_FOLDER --bert_name $BERT --model_path $SAVED_MODEL --split $SPLIT --seed $SEED

        echo vvvvvvvvvvvvvvv $COMM vvvvvvvvvvvvv >> ../logs/answer_fusion.log
        python fuse_optimizer.py --data_folder $DATA_FOLDER --model_path $SAVED_MODEL --mode 'base'
        echo ^^^^^^^^^^^^^^ $COMM ^^^^^^^^^^^^ >> ../logs/answer_fusion.log

        cd ..
    fi
done