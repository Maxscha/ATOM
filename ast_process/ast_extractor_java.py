import os
import argparse
import multiprocess as mp
import bash_exe as be
from tqdm import tqdm
import sys
from pygments.lexers import jvm
import json
import pandas as pd
import re
from nltk.corpus import wordnet
import nltk
from nltk.stem import WordNetLemmatizer

project_root = os.path.dirname(os.path.dirname(__file__))
SAVE_DIR = os.path.join(project_root, 'ast_process/data/java_relevant_data/raw_commits_from_github/')
FUNCTIONS_GENERATE_PATH = os.path.join(project_root, 'ast_process/data/java_relevant_data/functions_extracted_commits')
EXTRACTOR_CUSTOMIZED_JAR = os.path.join(project_root, 'JavaExtractor/JPredict/target/JavaExtractor-Customized.jar')
SAVE_AST_DIR = os.path.join(project_root, 'ast_process/data/java_relevant_data/ast_commits')
USED_DIFF_SAVED_PATH = os.path.join(project_root, 'ast_process/data/java_relevant_data/diffs_used_commits')
TOKENS_EXTRACTED_DIFFS = os.path.join(project_root, 'ast_process/data/java_relevant_data/tokens_extracted_diffs')
LETEX_FILTER_KEYWORDS = ['Token.Text', 'Token.Comment','Token.Operator']
PUNCTUATION_LIST = ['{}','?','',' ','.','/','!','(',')','[',']','?',',',';',':',"'",'"','-','{','}','...','‒','<<','>>','','\n']
FUNCTION_NAME_COMMIT_DICT = os.path.join(project_root, 'ast_process/function_name_commits.json')


def filter_functions(args):
    filter_project_commits = {}
    multi = args.multi
    total_count = 0
    if multi:
        projects = os.listdir(FUNCTIONS_GENERATE_PATH)
    else:
        projects = args.project
    for project in projects:
        project_path = os.path.join(FUNCTIONS_GENERATE_PATH, project)
        if os.path.isdir(project_path):
            files = os.listdir(project_path)
            filter = {}
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
            project_count = 0
            for commit_id in filter.keys():
                commit_file_obj = CommitFile(commit_id, filter[commit_id][0], filter[commit_id][1])
                if project not in filter_project_commits:
                    filter_project_commits[project] = [commit_file_obj]
                else:
                    filter_project_commits[project].append(commit_file_obj)
                project_count += 1
            print('project %s remaining commits %d' % (project, project_count))
            total_count += project_count
    print('all projects remain commits %d' % total_count)
    extract_ast_with_javaparser(filter_project_commits, multi)


def extract_ast_with_javaparser(filter_project_commits, multi):
    project_list = list(filter_project_commits.keys())
    commit_file_name = mp.Manager().dict()
    if multi:
        process_list = []
        for i in range(len(project_list)):
            process = mp.Process(target=single_process,
                                 args=(filter_project_commits[project_list[i]], project_list[i], commit_file_name))
            process_list.append(process)
            process.start()
        for process in process_list:
            process.join()
    else:
        for project in project_list:
            single_process(filter_project_commits[project], project, commit_file_name)

    with open(FUNCTION_NAME_COMMIT_DICT, 'w') as f:
        json.dump(dict(commit_file_name), f, indent=4)


def single_process(commit_file_obj_list, project, commit_file_name):
    commits_str = []
    function_used_commits = []
    commit_file_name_per_process = {}
    project_path = os.path.join(TOKENS_EXTRACTED_DIFFS + '/' + project)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    project_csv_data = read_csv_in_project(project)
    for i in tqdm(range(len(commit_file_obj_list)), file=sys.stdout, desc=project):
        commit_file_obj = commit_file_obj_list[i]
        commit_id = commit_file_obj.commit_id
        # if commit_id == '528115618aaac376b2b90ce39fd835e91ad1e492':
        #     print('ok')
        # else:
        #     continue
        function_used_commits.append(commit_id)
        if os.path.exists(SAVE_AST_DIR + '/' + project + '_ast.c2s'):
            exitsed_csv = pd.read_csv(SAVE_AST_DIR + '/' + project + '_ast.c2s')
            if commit_id in exitsed_csv['commit_id'].tolist():
                continue
        positive_functions_num = commit_file_obj.positive_functions_num
        negative_functions_num = commit_file_obj.negative_functions_num
        positive_restr = ''
        negative_restr = ''
        positive_function_used_diffs = []
        positive_dismatched_diffs = []
        negative_function_used_diffs = []
        negative_dismatched_diffs = []
        current_commit_msg, commit_file_name_per_process = generate_commit_msg(commit_id, project_csv_data, commit_file_name_per_process)
        if current_commit_msg is None or len(current_commit_msg) == 0:
            continue
        if positive_functions_num > 0:
            file_path = os.path.join(FUNCTIONS_GENERATE_PATH, project, project + '_' + commit_id + '_positive_' + str(positive_functions_num) + '.java')
            flag, positive_function_used_diffs, positive_dismatched_diffs = extract_used_diffs_into_tokens(commit_file_obj, project, 'positive')
            if flag:
                positive_restr = generate_single_function_ast(file_path, project)   # if token num = 1 cannot find path
                positive_restr = positive_restr.replace('\n', ' ')
        if negative_functions_num > 0:
            file_path = os.path.join(FUNCTIONS_GENERATE_PATH, project, project + '_' + commit_id + '_negative_' + str(negative_functions_num) + '.java')
            flag, negative_function_used_diffs, negative_dismatched_diffs = extract_used_diffs_into_tokens(commit_file_obj, project, 'negative')
            if flag:
                negative_restr = generate_single_function_ast(file_path, project)
                negative_restr = negative_restr.replace('\n', ' ')
        if positive_restr == '' and len(positive_function_used_diffs) == 0 and len(positive_dismatched_diffs) == 0 and \
                negative_restr == '' and len(negative_function_used_diffs) == 0 and len(negative_dismatched_diffs) == 0:
            continue
        current_commit_str = (commit_id, current_commit_msg, positive_restr, '\n'.join(positive_function_used_diffs),
                              '\n'.join(positive_dismatched_diffs), negative_restr, '\n'.join(negative_function_used_diffs), '\n'.join(negative_dismatched_diffs))
        commits_str.append(current_commit_str)
    if not os.path.exists(SAVE_AST_DIR + '/'):
        os.makedirs(SAVE_AST_DIR + '/')
    print('project %s has %d commits written into file' % (project, len(commits_str)))
    if len(commits_str) > 0:
        df = pd.DataFrame(commits_str, columns=['commit_id', 'commit_msg', 'positive_ast_paths', 'positive_function_used_diffs', 'positive_dismatched_diffs',
                                                'negative_ast_paths', 'negative_function_used_diffs', 'negative_dismatched_diffs'])
        if not os.path.exists(SAVE_AST_DIR + '/' + project + '_ast.c2s'):
            with open(SAVE_AST_DIR + '/' + '/' + project + '_ast.c2s', 'w') as f:
                df.to_csv(f, index=False)
        else:
            with open(SAVE_AST_DIR + '/' + project + '_ast.c2s', 'a') as f:
                df.to_csv(f, index=False)
    commit_file_name.update(commit_file_name_per_process)


def read_csv_in_project(project):
    project_path = os.path.join(SAVE_DIR, project + '_commits.csv')
    project_csv_data = pd.read_csv(project_path)
    return project_csv_data


def generate_commit_msg(commit_id, project_csv_data, commit_file_name_per_process):
    subject = project_csv_data[project_csv_data['commit_id'].isin([commit_id])]['subject']
    diff = project_csv_data[project_csv_data['commit_id'].isin([commit_id])]['diff']
    subject_str = str(subject.values[0]).lower()                       # normalization
    try:
        if subject_str == 'nan':
            return None, commit_file_name_per_process
        subject_str.encode(encoding='utf-8').decode('ascii')
    except:
        return None, commit_file_name_per_process
    diff_str = str(diff.values[0])
    file_names = get_diff_file_name(diff_str)
    if commit_id not in commit_file_name_per_process.keys():
        commit_file_name_per_process[commit_id] = list(set(file_names))
    return commit_msg_nlp_process(subject_str, file_names), commit_file_name_per_process


def get_diff_file_name(cur_diff):
    chunk_regex = r'diff --git'
    chuck_pattern = re.compile(chunk_regex)
    diff_chunks = chuck_pattern.split(cur_diff)
    file_names = []
    for chunk in diff_chunks:
        if chunk:
            lines = chunk.splitlines()
            for line in lines:
                if line.startswith('+++'):
                    if line.startswith('+++ b'):
                        file_names.append(line.split('+++ b')[-1].strip().split('/')[-1].lower().split('.')[0])
                    else:
                        file_names.append(line.split('+++ ')[-1].strip().split('/')[-1].lower().split('.')[0])
                if line.startswith('---'):
                    if line.startswith('--- a'):
                        file_names.append(line.split('--- a')[-1].strip().split('/')[-1].lower().split('.')[0])
                    else:
                        file_names.append(line.split('--- ')[-1].strip().split('/')[-1].lower().split('.')[0])
    return file_names


def commit_msg_nlp_process(msg, file_names):
    lemmatizer = WordNetLemmatizer()
    pattern = re.compile(r'\w+')                                # remove special characters
    words = pattern.findall(msg)
    new_words = []
    for word in words:
        if word in file_names:
            new_words.append('FILE')
        elif '_' in word:
            new_words.extend(word.split('_'))
        elif word.isdigit():
            new_words.append('NUMBER')
        else:
            new_words.append(word)
    lemmatizer_words_list = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in new_words if w.isalnum()]
    return '|'.join(commit_msg_coner_case(lemmatizer_words_list))


def commit_msg_coner_case(list):
    if len(list) >= 2:
        if list[0] == 'tika' and list[1] == 'NUMBER':                   # tika
            filter_list = list[2:]
        elif list[0] == 'ww' and list[1] == 'NUMBER':                   # struts
            filter_list = list[2:]
        elif list[0] == 'camel' and list[1] == 'NUMBER':                # camel
            filter_list = list[2:]
        elif list[0] == 'ca' and list[1] == 'NUMBER':                   # cas
            filter_list = list[2:]
        elif list[0] == 'hhh' and list[1] == 'NUMBER':                  # hibernate
            filter_list = list[2:]
        elif list[0] == 'spr' and list[1] == 'NUMBER':                  # spring-framework
            filter_list = list[2:]
        elif list[0] == 'sec' and list[1] == 'NUMBER':                  # spring-security
            filter_list = list[2:]
        else:
            filter_list = list
    else:
        filter_list = list
    return filter_list


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()                 # word type
    tag_dict = {'J': wordnet.ADJ,
                'N': wordnet.NOUN,
                'V': wordnet.VERB,
                'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def extract_used_diffs_into_tokens(commit_obj, project, type):
    lexer = jvm.JavaLexer()
    project_path = os.path.join(USED_DIFF_SAVED_PATH, project)
    with open(os.path.join(project_path, project + '_diff_' + commit_obj.commit_id + '_' + type + '.json')) as f:
        data = json.load(f)
        function_used_diffs = data['function_used_diffs']
        dismatched_diffs = data['dismatched_diffs']
        dictionary_list = []
        function_diffs = []
        for function_used_diff in function_used_diffs:
            function_name = function_used_diff[0]
            used_diff = function_used_diff[1]
            line_num = function_used_diff[2]
            function_start_num = function_used_diff[3]
            function_end_num = function_used_diff[4]
            if type == 'positive':
                used_diff = used_diff.replace('+', '').strip()
            if type == 'negative':
                used_diff = used_diff.replace('-', '').strip()
            used_diff = used_diff.strip()
            if used_diff == '/*' or used_diff == '*/' or \
                    len(used_diff) == 0 or used_diff == '/**' or used_diff == '**/':
                continue
            elif used_diff.startswith('*'):
                dismatched_diffs.append(used_diff)
            else:
                tokens = list(lexer.get_tokens(used_diff))
                filter_tokens = []
                for token in tokens:
                    flag = False
                    for keyword in LETEX_FILTER_KEYWORDS:
                        if keyword in str(token[0]):
                            if keyword == 'Token.Comment':
                                dismatched_diffs.append(str(token[1]))
                            flag = True
                            break
                    if token[1] in PUNCTUATION_LIST:
                        flag = True
                    if not flag:
                        filter_tokens.append(str(token[1]))
                if len(filter_tokens) != 0:
                    function_diffs.append(function_used_diff[1])
                    save_used_diff_info = {'function_name': function_name, 'raw_diff': used_diff, 'function_start_num': function_start_num,
                                           'function_end_num': function_end_num, 'line_num': line_num, 'tokens': filter_tokens, 'type': type}
                    dictionary_list.append(save_used_diff_info)
        dismatched_diffs = [dismatched_diff for dismatched_diff in dismatched_diffs if dismatched_diff.strip() not in ['+', '-']]
        if len(dictionary_list) > 0:
            file_path = os.path.join(TOKENS_EXTRACTED_DIFFS + '/' + project, project + '_' + commit_obj.commit_id + '_diff_tokens_' + type + '.json')
            with open(file_path, 'w') as f:
                json.dump(dictionary_list, f, indent=4)
            return True, function_diffs, dismatched_diffs
        else:
            return False, function_diffs, dismatched_diffs


def generate_single_function_ast(file_path, project):
    command = ['java', '-cp', EXTRACTOR_CUSTOMIZED_JAR, 'JavaExtractor.App', '--max_path_length', str(args.max_path_length), '--max_path_width',
              str(args.max_path_width), '--file', file_path]
    stdout = './' + project + '_stdout.txt'
    stderr = './' + project + '_stderr.txt'
    be.execute_command(command, '.', stdout, stderr)
    with open(os.path.join('tmp', project + "_stdout.txt")) as rfile:
        restr = rfile.read().strip('\n')
        return restr


class CommitFile:
    def __init__(self, commit_id, positive_functions_num, negative_functions_num):
        self.commit_id = commit_id
        self.positive_functions_num = positive_functions_num
        self.negative_functions_num = negative_functions_num


def test():
    lexer = jvm.JavaLexer()
    with open(os.path.join(project_root, 'ast_process/data/java_relevant_data/diffs_used_commits/cassandra/'
                                         'cassandra_diff_9ea61305ec30a476f48320c06f56d8d67000bbbe_positive.json')) as f:
        data = json.load(f)
        type = 'positive'
        function_used_diffs = data['function_used_diffs']
        dismatched_diffs = data['dismatched_diffs']
        function_diffs = []
        for function_used_diff in function_used_diffs:
            function_name = function_used_diff[0]
            used_diff = function_used_diff[1]
            line_num = function_used_diff[2]
            function_start_num = function_used_diff[3]
            function_end_num = function_used_diff[4]
            if type == 'positive':
                used_diff = used_diff.replace('+', '').strip()
            if type == 'negative':
                used_diff = used_diff.replace('-', '').strip()
            used_diff = used_diff.strip()
            if used_diff == '/*' or used_diff == '*/' or used_diff == '*' \
                    or len(used_diff) == 0 or used_diff == '/**' or used_diff == '**/':
                continue
            elif used_diff.startswith('*'):
                dismatched_diffs.append(used_diff)
            else:
                tokens = list(lexer.get_tokens(used_diff))
                filter_tokens = []
                for token in tokens:
                    flag = False
                    for keyword in LETEX_FILTER_KEYWORDS:
                        if keyword in str(token[0]):
                            if keyword == 'Token.Comment':
                                dismatched_diffs.append(token[1])
                            flag = True
                            break
                    if token[1] in PUNCTUATION_LIST:
                        flag = True
                    if not flag:
                        filter_tokens.append(str(token[1]))
                if len(filter_tokens) != 0:
                    function_diffs.append(used_diff)

        dismatched_diffs = [dismatched_diff for dismatched_diff in dismatched_diffs if dismatched_diff.strip() not in ['+','-']]
        print(dismatched_diffs)
        print(function_diffs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AST Generation from Java Functions')
    parser.add_argument('-p', '--project', nargs='+', help="Project Name")
    parser.add_argument("-m", "--multi", help="Specify if you want a single thread or multi-thread",
                        action="store_true")
    parser.add_argument('-l', '--max_path_length', help='clip AST by length', default=8)
    parser.add_argument('-w', '--max_path_width', help='clip AST by width', default=2)
    args = parser.parse_args()
    filter_functions(args)
    # test()