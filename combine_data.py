import html
import logging
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Union

import pandas as pd

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)

pd.options.mode.chained_assignment = None

def xml_to_dict(path: Union[str, os.PathLike]) -> List[Dict]:
    with open(path, 'r') as f:
        xml_root = ET.XML(f.read())

    return [{key: row.get(key) for key in row.keys()} for row in xml_root]


import re
# as per recommendation from @freylis, compile once only
CLEANR = re.compile('<.*?>')

def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    cleantext = html.unescape(cleantext)
    cleantext = cleantext.replace('\n','')
    return cleantext

def community_data(community):
    community_id = community.split('/')[-1]
    users = pd.DataFrame(xml_to_dict(os.path.join(community, 'Users.xml')))

    users = users.drop(['LastAccessDate', 'WebsiteUrl', 'Location','ProfileImageUrl'], axis=1)

    df_posts = pd.DataFrame(xml_to_dict(os.path.join(community, 'Posts.xml')))
    df_posts = df_posts.drop(['LastEditorUserId', 'LastEditDate',
                        'LastActivityDate', 'ContentLicense', 'ClosedDate',
                        'LastEditorDisplayName', 'OwnerDisplayName', 'CommunityOwnedDate'], axis=1)
    
    # df_posts = df_posts.dropna(subset=['OwnerUserId']) # deleted users
    df_posts = df_posts.merge(users[['Id', 'AccountId']], left_on='OwnerUserId', right_on='Id', suffixes=('','_'), how='left')
    df_posts['Id'] = df_posts['Id'].apply(lambda x: f'{community_id}_{x}')
    df_posts['AcceptedAnswerId'] = df_posts['AcceptedAnswerId'].apply(lambda x: x if pd.isna(x) else f'{community_id}_{x}')

    answered_post = df_posts.dropna(subset=['AcceptedAnswerId'])
    
    all_answers = df_posts[df_posts['PostTypeId'] == '2']
    all_answers['Text'] = all_answers['Body']
    all_answers['Text'] = all_answers['Text'].apply(cleanhtml)
    all_answers['ParentId'] = all_answers['ParentId'].apply(lambda x: f'{community_id}_{x}')
    all_answers = all_answers.drop(['AcceptedAnswerId', 'PostTypeId', 'Id_', 'AnswerCount', 'OwnerUserId', 'FavoriteCount', 'Body', 'ViewCount'], axis=1)
    
    all_questions = df_posts[df_posts['PostTypeId'] == '1']
    all_questions['Text'] = all_questions['Title'] + ' ' + all_questions['Body']
    all_questions['Text'] = all_questions['Text'].apply(cleanhtml)
    all_questions['Body'] = all_questions['Body'].apply(cleanhtml)
    all_questions['Title'] = all_questions['Title'].apply(cleanhtml)
    all_questions['Community'] = community_id
    # all_questions.dropna(subset=['OwnerUserId'])
    all_questions = all_questions.drop(['ParentId', 'PostTypeId', 'Id_', 'OwnerUserId'], axis=1)

    comments = pd.DataFrame(xml_to_dict(os.path.join(community, 'Comments.xml')))
    comments['Id'] = comments['Id'].apply(lambda x: f'{community_id}_{x}')
    comments['PostId'] = comments['PostId'].apply(lambda x: f'{community_id}_{x}')
    comments = comments.drop(['UserDisplayName'], axis=1)
    comments = comments.merge(users[['Id', 'AccountId']], left_on='UserId', right_on='Id', suffixes=('','_'))
    comments = comments.drop(['Id_', 'UserId'], axis=1)
    comments['Text'] = comments['Text'].apply(cleanhtml)
    
    postlinks = pd.DataFrame(xml_to_dict(os.path.join(community, 'PostLinks.xml')))[['PostId', 'RelatedPostId', 'LinkTypeId']]
    postlinks['PostId'] = postlinks['PostId'].apply(lambda x: f'{community_id}_{x}')
    postlinks['RelatedPostId'] = postlinks['RelatedPostId'].apply(lambda x: f'{community_id}_{x}')
    
    postlinks['LinkType'] = postlinks['LinkTypeId'].replace('1', 'related').replace('3', 'duplicated')
    postlinks = postlinks.drop('LinkTypeId', axis=1)

    tags = pd.DataFrame(xml_to_dict(os.path.join(community, 'Tags.xml')))
    tags['PostId'] = tags.ExcerptPostId.fillna(tags.WikiPostId)
    tags = tags.dropna()
    tags = tags.drop(['ExcerptPostId', 'WikiPostId'], axis=1)
    tags.PostId = tags.PostId.apply(lambda x: f'{community_id}_{x}')
    tags = tags.merge(df_posts[['Id', 'Body']], left_on='PostId', right_on='Id', suffixes=('', '_'))
    tags = tags.drop('Id_', axis=1)
    tags.Body = tags.Body.apply(cleanhtml)
    tags['Community'] = community_id
    
    return answered_post, all_answers, all_questions, users, comments, postlinks, tags

def stats_print(answered_post, all_answers, all_questions, users, comments):
    question_count = all_questions.shape[0]
    answered_count = answered_post.shape[0]
    atleast_one_answer_count = all_questions[all_questions.AnswerCount.apply(int) > 0].shape[0]
    answer_count = all_answers.shape[0]

    print('*'*5)
    print('Statistics:')
    print('*'*5)
    print(f'Total number of questions {question_count}')
    print(f'Total number of questions with atleast one answer {atleast_one_answer_count}')
    print(f'Total number of questions with an accepted answer {answered_count}')
    print(f'Total number of answers {answer_count}')


    all_user_text = pd.concat((all_questions[['Text', 'AccountId']], all_answers[['Text', 'AccountId']], comments[['Text', 'AccountId']]))
    user_count = all_user_text.groupby('AccountId')['Text'].count()
    print(f'Total Number of users in this community {len(user_count)}')
    print(f'Total Number of users with atleast 10 docs {len(user_count[user_count >= 10])}')



    print('-'*15)
def main():
    all_communities = [
        "writers",
        "workplace",
        "woodworking",
        "vegetarianism",
        "travel",
        "sustainability",
        "sports",
        "sound",
        "skeptics",
        "scifi",
        "rpg",
        "politics",
        "philosophy",
        "pets",
        "parenting",
        "outdoors",
        "opensource",
        "musicfans",
        "music",
        "movies",
        "money",
        "martialarts",
        "literature",
        "linguistics",
        "lifehacks",
        "law",
        "judaism",
        "islam",
        "interpersonal",
        "hsm",
        "history",
        "hinduism",
        "hermeneutics",
        "health",
        "genealogy",
        "gardening",
        "gaming",
        "freelancing",
        "fitness",
        "expatriates",
        "english",
        "diy",
        "cooking",
        "christianity",
        "buddhism",
        "boardgames",
        "bicycles",
        "apple",
        "anime",
        "academia"
    ]    
    # all_communities = ["writers"]
    all_data_dict = {
        'answered_post': [], 
        'all_answers': [], 
        'all_questions': [], 
        'users': [], 
        'comments': [], 
        'postlinks': [],
        'tags': []
        }
    all_communities = [os.path.join('raw_data', comm) for comm in all_communities]
    for comm in all_communities:
        # community_stats(comm)
        logging.info(f'Reading {comm}')
        answered_post, all_answers, all_questions, users, comments, postlinks, tags = community_data(comm)
        all_data_dict['answered_post'].append(answered_post)
        all_data_dict['all_answers'].append(all_answers)
        all_data_dict['all_questions'].append(all_questions)
        all_data_dict['users'].append(users)
        all_data_dict['comments'].append(comments)
        all_data_dict['postlinks'].append(postlinks)
        all_data_dict['tags'].append(tags)

    answered_post = pd.concat(pd.DataFrame(df) for df in all_data_dict['answered_post'])
    all_answers = pd.concat(pd.DataFrame(df) for df in all_data_dict['all_answers'])
    all_questions = pd.concat(pd.DataFrame(df) for df in all_data_dict['all_questions'])
    users = pd.concat(pd.DataFrame(df) for df in all_data_dict['users'])
    comments = pd.concat(pd.DataFrame(df) for df in all_data_dict['comments'])
    postlinks = pd.concat(pd.DataFrame(df) for df in all_data_dict['postlinks'])
    tags = pd.concat(pd.DataFrame(df) for df in all_data_dict['tags'])
    # stats_print(answered_post, all_answers, all_questions, users, comments)

    os.makedirs('dataset', exist_ok=True)
    answered_post.to_csv('dataset/questions_with_answer.csv', index=None)
    all_answers.to_csv('dataset/answers.csv', index=None)
    all_questions.to_csv('dataset/questions.csv', index=None)
    users.to_csv('dataset/users.csv', index=None)
    comments.to_csv('dataset/comments.csv', index=None)
    postlinks.to_csv('dataset/postlinks.csv', index=None)
    tags.to_csv('dataset/tags.csv', index=None)
    
if __name__ == '__main__':
    main()
