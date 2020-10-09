import pickle
from argparse import ArgumentParser
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
SAVE_AST_DIR = os.path.join('/home/shangqing/commit_data/raw_data/java_relevant_data/', 'ast_commits')
DATASET_DIR = '/home/shangqing/commit_data/training_data/translation'
RAW_DIR = os.path.join('/home/shangqing/commit_data/raw_data/java_relevant_data', 'raw_commits_from_github')
PUNCTUATION_LIST = ['+', '-', '*', '/', '{', '}', '.', ',', '(', ')', '[', ']', '?', '#', '@']


def process_file(data_frame, max_contexts, dataset_name, message_length):
    positive_path_list = data_frame['positive_ast_paths'].tolist()
    negative_path_list = data_frame['negative_ast_paths'].tolist()
    commit_id_list = data_frame['commit_id'].tolist()
    commit_msg_list = data_frame['commit_msg'].tolist()
    truncated_positive_list = truncated(positive_path_list, max_contexts)
    truncated_negative_list = truncated(negative_path_list, max_contexts)
    df = pd.DataFrame({'commit_id': commit_id_list, 'commit_msg': commit_msg_list,
                       'negative_path': truncated_negative_list, 'positive_path': truncated_positive_list},
                      columns=['commit_id', 'commit_msg', 'negative_path', 'positive_path'])
    df = shuffle(df)

    train_dataset, valid_dataset = train_test_split(df, test_size=0.1)
    train_dataset, test_dataset = train_test_split(train_dataset, test_size=0.1)

    print('train dataset shape:' + str(train_dataset.shape))
    print('valid dataset shape:' + str(valid_dataset.shape))
    print('test dataset shape:' + str(test_dataset.shape))
    train_dataset.to_csv(
        DATASET_DIR + '/' + dataset_name + '.' + str(max_contexts) + '.' + str(message_length) + '.train.c2s', index=False)
    valid_dataset.to_csv(
        DATASET_DIR + '/' + dataset_name + '.' + str(max_contexts) + '.' + str(message_length) + '.val.c2s', index=False)
    test_dataset.to_csv(
        DATASET_DIR + '/' + dataset_name + '.' + str(max_contexts) + '.' + str(message_length) + '.test.c2s', index=False)
    return train_dataset.shape[0]


def load_commit_date_dict():
    files = os.listdir(RAW_DIR)
    commit_date_dict = {}
    for file in files:
        try:
            df_project_reference = pd.read_csv(os.path.join(RAW_DIR, file))
            for index, row in df_project_reference.iterrows():
                commit_id = row['commit_id']
                date = row['date']
                commit_date_dict[commit_id] = date
        except:
            continue
    return commit_date_dict


def text_process(dismatched_diffs):
    lines = dismatched_diffs.split('\n')
    line_list = []
    for line in lines:
        for punction in PUNCTUATION_LIST:
            if punction in line:
                line = line.replace(punction, ' ')
        line = line.replace(';', ' ;')
        line = line.strip()
        if len(line) > 0 and line != ';':
            tokens = []
            for token in line.split():
                if token == '<PAD>':
                    continue
                else:
                    token = token.lower()
                tokens.append(token)
            line_list.append(' '.join(tokens))
    dismatched_diffs = ' '.join(line_list)
    if len(dismatched_diffs) == 0:
        dismatched_diffs = '<PAD>'
    return dismatched_diffs


def truncated(path_list, max_contexts):
    padded_list = []
    for positive_path in path_list:
        parts = positive_path.split()
        if len(parts) > max_contexts:
            parts = np.random.choice(parts, max_contexts, replace=False)
            padded_list.append(' '.join(parts))
        else:
            padded_list.append(positive_path)
    return padded_list


def read_subproject_in_directory():
    project_path = os.path.join(SAVE_AST_DIR)
    files = os.listdir(project_path)
    data_list = []
    for file in files:
        df_project = pd.read_csv(os.path.join(project_path, file))
        df_project['project'] = file
        data_list.append(df_project)
    df = pd.concat(data_list, ignore_index=True, sort=False)
    return df


def filter_df(df, args):
    commit_msg_list = df['commit_msg'].tolist()
    choose_index = []
    max_target_length = 0
    for i, commit_msg in enumerate(commit_msg_list):
        commit_msg_len = len(commit_msg.split('|'))
        if commit_msg_len > max_target_length:
            max_target_length = commit_msg_len
        if commit_msg_len <= int(args.message_lengths):
            choose_index.append(i)
        else:
            continue
    df = df.iloc[choose_index]
    print('after filter data size %s' % str(df.shape[0]))
    return df


def create_histograms(args):
    df = read_subproject_in_directory()
    print('raw data size %s' % str(df.shape[0]))
    df['positive_ast_paths'].fillna('<PAD>,<PAD>,<PAD>', inplace=True)
    df['negative_ast_paths'].fillna('<PAD>,<PAD>,<PAD>', inplace=True)
    df = filter_df(df, args)
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    coner_cases = []
    source_histogram, node_histogram, target_histogram = {}, {}, {}
    for index, row in df.iterrows():
        try:
            commit_msg = row['commit_msg']
            positive_path = row['positive_ast_paths']
            negative_path = row['negative_ast_paths']
            path = positive_path + ' ' + negative_path
            target_histogram = deal_with_component(commit_msg, target_histogram)
            source_histogram, node_histogram = deal_with_path(path, source_histogram, node_histogram)
        except:
            print(row['commit_id'])
            coner_cases.append(row['commit_id'])
    print('source vocab size is %d' % len(source_histogram.keys()))
    print('node vocab size is %d' % len(node_histogram.keys()))
    print('target vocab size is %d' % len(target_histogram.keys()))
    for histogram, data_role in zip([source_histogram, node_histogram, target_histogram], ['ori', 'node', 'tgt']):
        try:
            text = ''
            for k, v in zip(histogram.keys(), histogram.values()):
                text += str(k) + ' ' + str(v) + '\n'
            with open(os.path.join(DATASET_DIR, args.dataset_name + '.' + str(args.max_contexts) + '.' + str(args.message_lengths) + '.' + str(args.dismatched_lines) + '.histo.' + data_role + '.c2s'), 'w') as f:
                f.write(text)
        except:
            continue
    if len(coner_cases) > 0:
        print('coner case length: %d' % len(coner_cases))
        for commit_id in coner_cases:
            df.drop(df[df['commit_id'] == commit_id].index, inplace=True)
    return df, source_histogram, node_histogram, target_histogram


def deal_with_path(path, source_histogram, node_histogram):
    features = path.split()
    for feature in features:
        ori = feature.split(',')[0]
        nodes = feature.split(',')[1]
        des = feature.split(',')[2]
        source_histogram = deal_with_component(ori, source_histogram)
        node_histogram = deal_with_component(nodes, node_histogram)
        source_histogram = deal_with_component(des, source_histogram)
    return source_histogram, node_histogram


def deal_with_dismatched_diffs(line, histogram):
    for token in line.split(' '):
        if token in histogram.keys():
            histogram[token] += 1
        else:
            histogram[token] = 1
    return histogram


def deal_with_component(token, histogram):
    tokens = token.split('|')
    for token in tokens:
        if token in histogram.keys():
            histogram[token] += 1
        else:
            histogram[token] = 1
    return histogram


def save_dictionaries(dataset_name, subtoken_to_count, node_to_count, target_to_count, max_contexts, num_examples, message_length):
    with open(os.path.join(DATASET_DIR, dataset_name + '.' + str(max_contexts) + '.' + str(message_length) + '.dict.c2s'), 'wb') as file:
        pickle.dump(subtoken_to_count, file)
        pickle.dump(node_to_count, file)
        pickle.dump(target_to_count, file)
        pickle.dump(max_contexts, file)
        pickle.dump(num_examples, file)
        print('Dictionaries saved to: {}'.format(os.path.join(DATASET_DIR, dataset_name + '.' + str(max_contexts) +
                                                              '.' + str(message_length) + '.dict.c2s')))


def build_vocab(file_path):
    df = pd.read_csv(file_path)
    print('raw data size %s' % str(df.shape[0]))
    source_histogram, node_histogram, target_histogram = {}, {}, {}
    for index, row in df.iterrows():
        commit_msg = row['commit_msg']
        positive_path = row['positive_path']
        negative_path = row['negative_path']
        path = positive_path + ' ' + negative_path
        target_histogram = deal_with_component(commit_msg, target_histogram)
        source_histogram, node_histogram = deal_with_path(path, source_histogram, node_histogram)
    print('source vocab size is %d' % len(source_histogram.keys()))
    print('node vocab size is %d' % len(node_histogram.keys()))
    print('target vocab size is %d' % len(target_histogram.keys()))
    return list(source_histogram.keys()), list(node_histogram.keys()), list(target_histogram.keys())

def count_test(file_path, histogram):
    df = pd.read_csv(file_path)
    sample_num = 0
    sample_overlap = 0
    for index, row in df.iterrows():
        path = row['positive_path'] + ' ' + row['negative_path']
        features = path.split()
        commit_msg_tokens = row['commit_msg'].split('|')
        for feature in features:
            ori = feature.split(',')[0]
            des = feature.split(',')[2]
            ori_tokens = ori.split('|')
            des_tokens = des.split('|')
            flag = True
            for token in ori_tokens:
                if token not in histogram:
                    if token in commit_msg_tokens:
                        sample_num += 1
                        flag = False
                        break
            if flag:
                for token in des_tokens:
                    if token not in histogram:
                        if token in commit_msg_tokens:
                            sample_num += 1
                            break
        for feature in features:
            flag_overlap = False
            ori = feature.split(',')[0]
            des = feature.split(',')[2]
            ori_tokens = ori.split('|')
            des_tokens = des.split('|')
            for token in ori_tokens + des_tokens:
                if token in commit_msg_tokens:
                    flag_overlap = True
                    sample_overlap += 1
                    break
            if flag_overlap:
                break
    print('%d samples have tokens from diff' % sample_num)
    print('%d samples have overlaps' % sample_overlap)


def count_test_1(file_path, histogram):
    df = pd.read_csv(file_path)
    count = 0
    for index, row in df.iterrows():
        path = row['positive_path'] + ' ' + row['negative_path']
        features = path.split()
        for feature in features:
            flag = False
            ori = feature.split(',')[0]
            des = feature.split(',')[2]
            ori_tokens = ori.split('|')
            des_tokens = des.split('|')
            for token in ori_tokens + des_tokens:
                if token not in histogram:
                    count += 1
                    flag = True
                    break
            if flag:
                break
    print('%d samples have tokens not from vocab set' % count)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset_name", default='commit2seq', help="the name of the dataset constructed", required=False)
    parser.add_argument("-m", "--max_contexts", default=100, help="number of max contexts to keep in the dataset", required=False)
    parser.add_argument('-l', '--message_lengths', help='Filter out too long messages', default=20)
    args = parser.parse_args()
    df, source_histogram, node_histogram, target_histogram = create_histograms(args)

    node_to_count = dict([(k, node_histogram[k]) for k in sorted(node_histogram, key=node_histogram.get, reverse=True)])
    subtoken_to_count = dict(
        [(k, source_histogram[k]) for k in sorted(source_histogram, key=source_histogram.get, reverse=True)])
    target_to_count = dict(
        [(k, target_histogram[k]) for k in sorted(target_histogram, key=target_histogram.get, reverse=True)])
    num_examples = process_file(data_frame=df, max_contexts=int(args.max_contexts), dataset_name=args.dataset_name, message_length=args.message_lengths)

    save_dictionaries(dataset_name=args.dataset_name, subtoken_to_count=subtoken_to_count,
                      node_to_count=node_to_count, target_to_count=target_to_count, max_contexts=int(args.max_contexts),
                      num_examples=num_examples, message_length=args.message_lengths)

