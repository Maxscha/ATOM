import pandas as pd
import multiprocess as mp
import re
from git import *
import bash_exe as be
import argparse
from tqdm import tqdm
import sys
import os
import json
project_root = os.path.dirname(os.path.dirname(__file__))
CONSTANT_REPO_PATH = '/home/shangqing/Downloads/project/java-project'
RAW_DATA_PATH = os.path.join(project_root, 'ast_process/data/java_relevant_data/raw_commits_from_github')
FUNCTIONS_GENERATE_PATH = os.path.join(project_root, 'ast_process/data/java_relevant_data/functions_extracted_commits')
USED_DIFF_SAVED_PATH = os.path.join(project_root, 'ast_process/data/java_relevant_data/diffs_used_commits')
FILTER_DIFFS_NUM = 10
FILTER_CHUNKS_NUM = 5


def get_commits(raw_project_commits, from_dir):
    project_commits = dict()
    if from_dir:
        for raw_project in raw_project_commits:
            commit_objs = []
            project_path = os.path.join(RAW_DATA_PATH, raw_project + '_commits.csv')
            df = pd.read_csv(project_path)
            df = df.dropna(axis=0, subset=['subject'])
            df = df.dropna(axis=0, subset=['diff'])
            commits = zip(list(df['commit_id']), list(df['parent_commit']), list(df['diff']))
            for commit in list(commits):
                commit_obj = Commit(commit[0], commit[1], commit[2])
                commit_objs.append(commit_obj)
            project_commits[raw_project] = commit_objs
    else:
        commit_objs = []
        project_path = os.path.join(RAW_DATA_PATH, raw_project_commits + '_commits.csv')
        df = pd.read_csv(project_path)
        df = df.dropna(axis=0, subset=['subject'])
        df = df.dropna(axis=0, subset=['diff'])
        commits = zip(list(df['commit_id']), list(df['parent_commit']), list(df['diff']))
        for commit in list(commits):
            commit_obj = Commit(commit[0], commit[1], commit[2])
            commit_objs.append(commit_obj)
        project_commits[raw_project_commits] = commit_objs
    print('raw project commits load successful')
    return project_commits


def generate_functions(args):
    multiprocess = args.multi
    projects = args.project
    is_server = args.server
    if multiprocess:
        if not is_server:
            project_commits = get_commits(projects, from_dir=True)
        else:
            projects = []
            project_paths = os.listdir(RAW_DATA_PATH)
            for project_path in project_paths:
                project_name = project_path.split('.')[0].split('_')[0]
                projects.append(project_name)
            project_commits = get_commits(projects, from_dir=True)
        process_list = []
        project_list = list(project_commits.keys())
        for i in range(len(project_list)):
            process = mp.Process(target=single_process,
                                 args=(project_commits[project_list[i]], project_list[i]))
            process_list.append(process)
            process.start()
        for process in process_list:
            process.join()
    else:
        single_project = projects[0]
        project_commits = get_commits(single_project, from_dir=False)
        single_process(project_commits[single_project], single_project)


def single_process(commit_objs, project):
    print('begin project %s' % project)
    project_path = os.path.join(args.repo_dir, project)
    if not os.path.exists(project_path):
        print("Repo does not exists: %s" % project_path)
    else:
        if not os.path.exists(os.path.join(FUNCTIONS_GENERATE_PATH, project)):
            os.makedirs(os.path.join(FUNCTIONS_GENERATE_PATH, project))
        if not os.path.exists(os.path.join(USED_DIFF_SAVED_PATH, project)):
            os.makedirs(os.path.join(USED_DIFF_SAVED_PATH, project))
        existed_diffs = os.listdir(os.path.join(USED_DIFF_SAVED_PATH + '/' + project))
        existed_commits = list(set([existed_diff.split('_')[2] for existed_diff in existed_diffs]))
        for i in tqdm(range(len(commit_objs)), file=sys.stdout, desc=project):
            commit = commit_objs[i]
            commit_id = commit.hash_id
            if commit_id in existed_commits:
                continue
            try:
                positive_functions, positive_matched_diffs, positive_dismatched_diffs = [], [], []
                negative_functions, negative_matched_diffs, negative_dismatched_diffs = [], [], []
                chunk_obj_list = process_single_diff(commit_id, commit.diff)
                if len(chunk_obj_list) == 0:
                    continue
                for chunk_obj in chunk_obj_list:
                    for change_obj in chunk_obj.changes:
                        negative_changed_file = chunk_obj.negative_changed_file
                        positive_changed_file = chunk_obj.positive_changed_file
                        if negative_changed_file.endswith('.java') or positive_changed_file.endswith('.java'):     # remove contain non .java file
                            positive_functions, positive_matched_diffs, positive_dismatched_diffs = get_function_from_commit(commit.hash_id, positive_changed_file, project_path,  change_obj.positive_contexts, project, positive_functions, positive_matched_diffs, positive_dismatched_diffs)
                            negative_functions, negative_matched_diffs, negative_dismatched_diffs = get_function_from_commit(commit.parent_id, negative_changed_file, project_path,  change_obj.negative_contexts, project, negative_functions, negative_matched_diffs, negative_dismatched_diffs)
                        else:
                            continue
                negative_count = len(negative_functions)
                positive_count = len(positive_functions)
                if negative_count > 0 and positive_count > 0:
                    write_func(negative_functions, commit.hash_id, project, FUNCTIONS_GENERATE_PATH, negative_count, type='negative')
                    write_used_diffs(negative_matched_diffs, negative_dismatched_diffs, commit.hash_id, project, USED_DIFF_SAVED_PATH, type='negative')
                    write_func(positive_functions, commit.hash_id, project, FUNCTIONS_GENERATE_PATH, positive_count, type='positive')
                    write_used_diffs(positive_matched_diffs, positive_dismatched_diffs, commit.hash_id, project, USED_DIFF_SAVED_PATH, type='positive')
            except:
                print('commit: %s is invalid' % commit.hash_id)
        print('project: %s function generator finished' % project)


def process_single_diff(commit_id, cur_diff):
    chunk_regex = r'diff --git'
    change_regex = r'@@\s-.*,.*\+.*,.*\s@@'
    chuck_pattern = re.compile(chunk_regex)
    change_pattern = re.compile(change_regex)
    diff_chunks = chuck_pattern.split(cur_diff)
    chunk_obj_list = []
    if len(diff_chunks) > FILTER_CHUNKS_NUM:
        return []
    for chunk in diff_chunks:
        if chunk:
            chunk_obj = Chunk(chunk)
            changes = change_pattern.split(chunk)
            if len(changes) > FILTER_DIFFS_NUM:
                return []
            change_obj_list = []
            line_at_list = re.findall(change_regex, chunk)
            for i, change in enumerate(changes):
                if i == 0:
                    lines = chunk.splitlines()
                    for line in lines:
                        if line.startswith('+++'):
                            if line.startswith('+++ b'):
                                chunk_obj.positive_changed_file = line.split('+++ b')[-1].strip()
                            else:
                                chunk_obj.positive_changed_file = line.split('+++ ')[-1].strip()
                        if line.startswith('---'):
                            if line.startswith('--- a'):
                                chunk_obj.negative_changed_file = line.split('--- a')[-1].strip()
                            else:
                                chunk_obj.negative_changed_file = line.split('--- ')[-1].strip()
                else:
                    negative_start = int(line_at_list[i-1].split(' ')[1].split(',')[0].split('-')[1])
                    positive_start = int(line_at_list[i-1].split(' ')[2].split(',')[0].split('+')[1])
                    contexts = change.split('\n')
                    function_name_part_info = contexts[0]
                    negative_contexts = []
                    positive_contexts = []
                    positive_count = 0
                    negative_count = 0
                    for context in contexts[1:len(contexts)-1]:
                        if context.startswith('-'):
                            negative_contexts.append((context, negative_count + negative_start))
                            negative_count += 1
                        elif context.startswith('+'):
                            positive_contexts.append((context, positive_count + positive_start))
                            positive_count += 1
                        else:
                            # negative_contexts.append((context, negative_count + negative_start))
                            # positive_contexts.append((context, positive_count + positive_start))
                            negative_count += 1
                            positive_count += 1
                    if negative_start != 0:
                        negative_start += 3                      # find true difference line
                    if positive_start != 0:
                        positive_start += 3                      # find true difference line
                    change_obj = Change(change, negative_start, positive_start, function_name_part_info, negative_contexts, positive_contexts)
                    change_obj_list.append(change_obj)
            chunk_obj.add_changes(change_obj_list)
            chunk_obj_list.append(chunk_obj)
    return chunk_obj_list


def get_function_from_commit(commit, function_directory, project_path, contexts, project, functions, matched_diffs, dismatched_diffs):
    if len(contexts) == 0:
        return functions, matched_diffs, dismatched_diffs
    tmp_file_path = get_commit_file_path(commit, function_directory, project_path, project)
    func_list = get_funcs_in_file(tmp_file_path, commit, project_path, project)
    matched_records = []
    for func_tup in func_list:
        start_line = func_tup['start_line']
        end_line = func_tup['end_line']
        function_name = func_tup['function_name']
        for context in contexts:
            if start_line <= context[1] <= end_line:
                if func_tup['func'] not in functions:
                    functions.append(func_tup['func'])
                single_diff = str(context[0]).replace('\n', '').replace('\r', '')
                matched_diffs.append((function_name, single_diff, str(context[1]), str(start_line), str(end_line)))
                matched_records.append(context[0])
    dismatched_diffs.extend([str(ele).replace('\n', '').replace('\r', '') for ele in list(set([context[0] for context in contexts]) - set(matched_records))])
    return functions, matched_diffs, dismatched_diffs


def get_commit_file_path(commit_id, file_path, repo_path, project):
    file_path = '.' + file_path
    cmd = ['git', 'show', commit_id + ':' + file_path]
    last_name = file_path.split(os.sep)[-1]
    stdout = './' + project + '_stdout.txt'
    stderr = './' + project + '_stderr.txt'
    be.execute_command(cmd, repo_path, stdout, stderr)
    output = os.path.join('tmp', project + '_stdout.txt')
    with open(output, 'r') as rfile:
        old_file_contents = rfile.read()
    try:
        tmp_file_path = os.path.join('tmp', last_name)
        with open(tmp_file_path, 'w') as wfile:
            wfile.write(old_file_contents)
    except IOError:
        return ""

    return tmp_file_path


def get_funcs_in_file(file_path, sha_id, project_path, project):
    func_dict_list = []
    cmd2 = ['ctags', '--fields=+ne-t', '-o -', '--sort=no', '--excmd=number', file_path]
    stdout = './' + project + '_stdout.txt'
    stderr = './' + project + '_stderr.txt'
    be.execute_command(cmd2, ".", stdout, stderr)
    with open(os.path.join('tmp', project + "_stdout.txt")) as rfile:
        res = rfile.read()

    if not res:
        with open(file_path) as rfile:
            file_str = rfile.read()
        if not file_str:
            if os.path.exists(file_path):
                os.remove(file_path)
            return None
        else:
            print("Empty Ctags Result [%s/%s]" % (project_path, sha_id))
    lines = res.splitlines()

    for line in lines:
        fields = line.split()
        func_dict = {}
        if 'm' in fields or 'f' in fields:           # add function or identifier
            start_num = get_num(fields, 'line:')
            end_num = get_num(fields, 'end:')
            func_dict['func'] = extract_function(file_path, start_num, end_num)
            func_dict['start_line'] = start_num
            func_dict['end_line'] = end_num
            func_dict['function_name'] = fields[0]
            func_dict_list.append(func_dict)
    if os.path.exists(file_path):
        os.remove(file_path)
    return func_dict_list


def extract_function(file_path, start_num, end_num):
    with open(file_path) as rfile:
        text = rfile.read()
        lines = text.split('\n')
        return "\n".join(lines[start_num-1:end_num])


def get_num(fields, tag):
    try:
        for item in fields:
            if tag in item:
                tag_list = item.split(":")
                return int(tag_list[-1])
    except:
        print(fields, tag)
        input()


def write_func(functions, commit_id, project, functions_generator_path, count, type):
    file_name = project + '_' + commit_id + '_' + type + '_' + str(count) + '.java'
    function_path = os.path.join(functions_generator_path + '/' + project, file_name)
    funcstr = '\n'.join(functions)
    with open(function_path, 'w') as w:
        w.write(funcstr)


def write_used_diffs(used_diffs, dismatched_diffs, commit_id, project, diffs_generator_path, type):
    file_name = project + '_diff_' + commit_id + '_' + type + '.json'
    diff_used_path = os.path.join(diffs_generator_path + '/' + project, file_name)
    with open(diff_used_path, 'w') as w:
        json.dump({'function_used_diffs': used_diffs, 'dismatched_diffs': dismatched_diffs}, w)


class Commit:
    def __init__(self, hash_id, parent_id, diff):
        self.hash_id = hash_id
        self.parent_id = parent_id
        self.diff = diff


class Chunk:
    def __init__(self, chunk_str):
        self.chunk_str = chunk_str
        self.negative_changed_file = ''
        self.positive_changed_file = ''
        self.changes = []

    def add_changes(self, change_obj_list):
        self.changes = change_obj_list


class Change:
    def __init__(self, change_str, negative_start, positive_start, function_name_part_info, negative_contexts, positive_contexts):
        self.change_str = change_str
        self.negative_start = negative_start
        self.positive_start = positive_start
        self.function_name_part_info = function_name_part_info
        self.negative_contexts = negative_contexts
        self.positive_contexts = positive_contexts


def test():
    stdout = './testout.txt'
    stderr = './testerr.txt'
    file_path = '/java/org/apache/jasper/compiler/JDTCompiler.java'
    file_path = '.' + file_path
    commit_id = '455b6f8532fa673acc06b57b61279d5a2c4b6887'
    cmd = ['git', 'show', commit_id + ':' + file_path]
    cmd2 = ['ctags', '--fields=+ne-t', '-o -', '--sort=no', '--excmd=number', file_path]
    be.execute_command(cmd, '/home/shangqing/Downloads/project/java-project/tomcat/', stdout, stderr)
    # be.execute_command(cmd2, ".", stdout, stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate functions from Commits')
    parser.add_argument("-p", '--project', nargs='+', help="Project Name")
    parser.add_argument("-m", "--multi", help="Specify if you want a single thread or multi-thread",
                        action="store_true")
    parser.add_argument("-s", "--server", help="Specify run on the server", action="store_true")
    parser.add_argument('-r', '--repo_dir', default=CONSTANT_REPO_PATH, help="Projects dir need to preprocess")
    args = parser.parse_args()
    generate_functions(args)
    # test()