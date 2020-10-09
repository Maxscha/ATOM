import os
import pandas as pd
project_root = '/home/shangqing/Documents/GitHub/c2m'
FUNCTIONS_GENERATE_PATH = os.path.join(project_root, 'ast_process/data/java_relevant_data/functions_extracted_commits')
SAVE_AST_DIR = os.path.join(project_root, 'ast_process/data/java_relevant_data/ast_commits')
import numpy as np
import matplotlib.pyplot as plt


def analyse_raw_dataset():
    files = os.listdir(os.path.join(project_root, 'ast_process/data/java_relevant_data/raw_commits_from_github/'))
    total_rows = 0
    for file in files:
        file_path = os.path.join(os.path.join(project_root, 'ast_process/data/java_relevant_data/raw_commits_from_github/'), file)
        df = pd.read_csv(file_path)
        total_rows += int(df.shape[0])
    print(total_rows)


def count_valid_commits():
    filter = {}
    projects = os.listdir(FUNCTIONS_GENERATE_PATH)
    for project in projects:
        project_path = os.path.join(FUNCTIONS_GENERATE_PATH, project)
        files = os.listdir(project_path)
        for file_name in files:
            commit_id = file_name.split('_')[1]
            type = file_name.split('_')[2]
            num = int(file_name.split('_')[-1].split('.')[0])
            if commit_id not in filter:
                if type == 'positive':
                    filter[commit_id] = [num, 0]
                if type == 'negative':
                    filter[commit_id] = [0, num]
            else:
                if type == 'positive':
                    filter[commit_id][0] = num
                if type == 'negative':
                    filter[commit_id][1] = num
    zero_count = 0
    non_zero_count = 0
    print(len(list(filter.keys())))
    for commit_id in filter.keys():
        if filter[commit_id][0] == 0 and filter[commit_id][1] == 0:
            zero_count += 1
        else:
            non_zero_count += 1
    print('zeros count: %d ' % zero_count)
    print('non zero count: %d ' % non_zero_count)


def read_subproject_in_directory():
    files = os.listdir(SAVE_AST_DIR)
    data_list = []
    for file in files:
        df_project = pd.read_csv(os.path.join(SAVE_AST_DIR, file))
        data_list.append(df_project)
    df = pd.concat(data_list, ignore_index=True, sort=False)
    return df


def analyse_dataset():
    dataframe = read_subproject_in_directory()
    positive_path_list = dataframe['positive_ast_paths'].tolist()
    negative_path_list = dataframe['negative_ast_paths'].tolist()
    commit_msg_list = dataframe['commit_msg'].tolist()
    # analyse_commit_msg(commit_msg_list)
    # print('-------------------------------------------------------------------')
    analyse(positive_path_list)
    print('-------------------------------------------------------------------')
    analyse(negative_path_list)


def analyse_commit_msg(commit_msg_list):
    msg_num_list = []
    for commit_msg in commit_msg_list:
        if isinstance(commit_msg, str):
            msg_num_list.append(len(commit_msg.split('|')))
    bins = [0, 2, 4, 6, 8, 10, 12, 15, 20, 30, 50]
    token_len = pd.DataFrame(msg_num_list, columns=['range'])
    cats = pd.cut(token_len.range, bins)
    grouped = token_len.groupby(cats)['range'].count().to_frame().rename(
        columns={'range': 'commit_msg'}).reset_index()
    print(grouped)


def analyse(positive_path_list):
    contexts_num_list = []
    sources_num_list = []
    dest_num_list = []
    paths_num_list = []
    for positive_path in positive_path_list:
        if isinstance(positive_path, str):
            parts = positive_path.split()
            contexts_num_list.append(len(parts))
            for part in parts:
                source = part.split(',')[0]
                paths = part.split(',')[1]
                dest = part.split(',')[2]
                sources_num_list.append(len(source.split('|')))
                dest_num_list.append(len(dest.split('|')))
                paths_num_list.append(len(paths.split('|')))
    bins = [0, 10, 25, 50, 80, 100, 200, 300, 400, 500, 1000]
    token_len = pd.DataFrame(contexts_num_list, columns=['range'])
    cats = pd.cut(token_len.range, bins)
    grouped = token_len.groupby(cats)['range'].count().to_frame().rename(
        columns={'range': 'context-num'}).reset_index()
    print(grouped)
    print('mean num of paths: %f' % (sum(contexts_num_list)/float(len(positive_path_list))))

    # bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # token_len = pd.DataFrame(paths_num_list, columns=['range'])
    # cats = pd.cut(token_len.range, bins)
    # grouped = token_len.groupby(cats)['range'].count().to_frame().rename(
    #     columns={'range': 'path'}).reset_index()
    # print(grouped)
    #
    # bins = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 25, 50]
    # token_len = pd.DataFrame(sources_num_list, columns=['range'])
    # cats = pd.cut(token_len.range, bins)
    # grouped = token_len.groupby(cats)['range'].count().to_frame().rename(
    #     columns={'range': 'source'}).reset_index()
    # print(grouped)
    #
    # bins = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 25, 50]
    # token_len = pd.DataFrame(dest_num_list, columns=['range'])
    # cats = pd.cut(token_len.range, bins)
    # grouped = token_len.groupby(cats)['range'].count().to_frame().rename(
    #     columns={'range': 'des'}).reset_index()
    # print(grouped)


if __name__ == '__main__':
    # analyse_dataset()
    # count_valid_commits()
    analyse_raw_dataset()