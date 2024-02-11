python create_best_answer_data.py --dataset_folder ../dataset --train_split_time '2019-12-31 23:59:59' --test_split_time '2020-12-31 23:59:59'

python optimize_bm25.py --dataset_folder ../dataset/answer_retrieval --cpus 10 --index_name stack_answers --ip localhost --port 9200 --mapping_path mapping.json --top_k 100 --val_size 2000 --seed 42

python get_bm25_runs.py --dataset_folder ../dataset/answer_retrieval --cpus 10 --index_name stack_answers --ip localhost --port 9200 --mapping_path mapping.json --train_top_k 100 --val_top_k 100 --test_top_k 100

# run this if there are some errors in the previous run due to timeout issues
# python get_remaining_bm25_runs.py --dataset_folder ../dataset/answer_retrieval --index_name stack_answers --ip localhost --port 9200 --mapping_path mapping.json --train_top_k 100 --val_top_k 100 --test_top_k 100

python create_final_run.py --dataset_folder ../dataset/answer_retrieval --train_top_k 100 --val_top_k 100 --test_top_k 100
