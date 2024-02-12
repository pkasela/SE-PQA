import json
from os import makedirs
from os.path import join

import click
import pandas as pd
from tqdm import tqdm


def create_and_save_collection(df, out_name):
    collection = {row['Id']: row['Text'] for id, row in tqdm(df.iterrows(), 
                                                             desc='Creating Collection', 
                                                             total=df.shape[0])}
    with open(out_name, 'w') as f:
        json.dump(collection, f, indent=2)


def train_val_test_split(df, train_split_time, test_split_time):
    df_train = df[df.CreationDate < train_split_time]
    df_val = df[(df.CreationDate >= train_split_time) & (df.CreationDate < test_split_time)]
    df_test = df[df.CreationDate >= test_split_time]
    # remove (for me 1044) questions with no user id, so the future user can decide to use or not use user data for testing
    df_test = df_test[~df_test.isna()] 

    return df_train, df_val, df_test

def create_data_jsonl(data_df, out_name, df_question, df_answers, test_split_time):
    data_jsonl = []
    pers_data_jsonl = []
    for _, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):
        earliest_data = min(row['CreationDate'], test_split_time)

        user_questions = [] if pd.isna(row['AccountId']) else list(df_question[(df_question.AccountId == int(row['AccountId'])) & (df_question.CreationDate < earliest_data)].Id)
        user_answers = [] if pd.isna(row['AccountId']) else list(df_answers[(df_answers.AccountId == int(row['AccountId'])) & (df_answers.CreationDate < earliest_data)].Id)

        if len(user_questions) >= 5 and pd.notna(row['AcceptedAnswerId']): 
            pers_data_jsonl.append(
                {
                    'id': row['Id'], 
                    'text': row['Text'], 
                    'title': row['Title'],
                    'timestamp': row['CreationDate'],
                    'score': row['Score'],
                    'views': row['ViewCount'],
                    'favorite': row['FavoriteCount'],
                    'user_id': int(row['AccountId']) if not pd.isna(row['AccountId']) else -1,
                    'user_questions': user_questions,
                    'user_answers': user_answers,
                    'tags': row['Tags'].strip('<').strip('>').split('><'),
                    'rel_ids': [row['AcceptedAnswerId']], 
                    'rel_scores': [1], 
                    'rel_timestamps': row['Timestamps'],
                    'best_answer': row['AcceptedAnswerId']
                } 
            )
        data_jsonl.append(
            {
                'id': row['Id'], 
                'text': row['Text'], 
                'title': row['Title'],
                'timestamp': row['CreationDate'],
                'score': row['Score'],
                'views': row['ViewCount'],
                'favorite': row['FavoriteCount'],
                'user_id': int(row['AccountId']) if not pd.isna(row['AccountId']) else -1,
                'user_questions': user_questions,
                'user_answers': user_answers,
                'tags': row['Tags'].strip('<').strip('>').split('><'),
                'rel_ids': row['AnswerIds'], 
                'rel_scores': row['Scores'], 
                'rel_timestamps': row['Timestamps'],
                'best_answer': row['AcceptedAnswerId']
            } 
        )

    with open(out_name, 'w') as f:
        for row in tqdm(data_jsonl, desc='Writing jsonl'):
                json.dump(row, f)
                f.write('\n')
                
                
    with open(out_name.replace('.json', '_pers.json'), 'w') as f:
        for row in tqdm(pers_data_jsonl, desc='Writing pers jsonl'):
                json.dump(row, f)
                f.write('\n')

@click.command()
@click.option(
    "--dataset_folder",
    type=str,
    required=True,
)
@click.option(
    "--train_split_time",
    type=str,
    required=True,
)
@click.option(
    "--test_split_time",
    type=str,
    required=True,
)
def main(dataset_folder, train_split_time, test_split_time):
    # dataset_folder = '../dataset'
    df_question = pd.read_csv(join(dataset_folder, 'questions.csv'), lineterminator='\n')
    df_question.CreationDate = pd.to_datetime(df_question.CreationDate).apply(lambda x: int(x.timestamp())) 
        
    df_answers = pd.read_csv(join(dataset_folder, 'answers.csv'), lineterminator='\n')
    df_answers.CreationDate = pd.to_datetime(df_answers.CreationDate).apply(lambda x: int(x.timestamp())) 


    makedirs(join(dataset_folder, 'answer_retrieval'), exist_ok=True)

    df_answers = df_answers[df_answers['Score'] >= 0] # remove negative samples from data
    create_and_save_collection(df_answers, join(dataset_folder, 'answer_retrieval/answer_collection.json'))
    create_and_save_collection(df_question, join(dataset_folder, 'answer_retrieval/question_collection.json'))

    # df_answers = df_answers[df_answers['Score'] >= 0] # remove negative samples from data
    df_ans_question = df_answers.groupby('ParentId').agg({
                                                            'Id': list, 
                                                            'Score': list, 
                                                            'CreationDate': list
                                                        }).reset_index()
    df_ans_question.columns = ['QuestionId', 'AnswerIds', 'Scores', 'Timestamps']

    df_ans_question = df_ans_question.merge(df_question[[
                                                         'Id', 'Text', 'Title', 'CreationDate', 
                                                         'AccountId', 'Tags', 'AcceptedAnswerId', 
                                                         'Score', 'ViewCount', 'FavoriteCount'
                                                        ]], 
                                            left_on='QuestionId', right_on='Id') 


    train_split_time = int(pd.to_datetime(train_split_time).timestamp()) # '2019-12-31 23:59:59'
    test_split_time = int(pd.to_datetime(test_split_time).timestamp()) # '2020-12-31 23:59:59'
    df_ans_train, df_ans_val, df_ans_test = train_val_test_split(df_ans_question, train_split_time, test_split_time)

    makedirs(join(dataset_folder, 'answer_retrieval/train'), exist_ok=True)
    makedirs(join(dataset_folder, 'answer_retrieval/val'), exist_ok=True)
    makedirs(join(dataset_folder, 'answer_retrieval/test'), exist_ok=True)

    create_data_jsonl(df_ans_train, join(dataset_folder, 'answer_retrieval/train/data.jsonl'), df_question, df_answers, test_split_time)
    create_data_jsonl(df_ans_val, join(dataset_folder, 'answer_retrieval/val/data.jsonl'), df_question, df_answers, test_split_time)
    create_data_jsonl(df_ans_test, join(dataset_folder, 'answer_retrieval/test/data.jsonl'), df_question, df_answers, test_split_time)
    
    
if __name__ == '__main__':
    main()