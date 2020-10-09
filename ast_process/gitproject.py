from git import *
import csv
from tqdm import tqdm
import sys
import os
import multiprocess as mp
project_root = os.path.dirname(os.path.dirname(__file__))
PROJECTS_DIR = "/home/shangqing/Downloads/project/java-project"
SAVE_DIR = os.path.join(project_root, 'ast_process/data/java_relevant_data/raw_commits_from_github/')


class GitProject:
    def __init__(self, local_project_name=None, repo_path=None):
        if repo_path is not None:
            self.repo_path = repo_path
        else:
            local_repo_path = os.path.join(args.project_dir, local_project_name)
            if os.path.exists(local_repo_path):
                self.repo_path = local_repo_path
            else:
                print("[GitProject] There is no such repository in the local storage file")
                exit(1)
        self.project_name = local_project_name

    def process_commits(self, statics):
        repo = Repo(self.repo_path)
        commits_obj = list(repo.iter_commits(reverse=True))
        commits_list = []
        for commit in commits_obj:
            commits_list.append(str(commit))
        self.single_process_commits(commits_list, repo, statics)

    def single_process_commits(self, commits_list, repo, statics):
        merge_num = 0
        message_null = 0
        non_ascill = 0
        commit_obj_list_filter = []
        for i in tqdm(range(len(commits_list)), file=sys.stdout, disable=False):
            commit = GitCommit(commits_list[i], repo, self.project_name)
            if commit.commit_msg is None:
                non_ascill += 1
            elif commit.commit_msg.startswith('Merge'):
                merge_num += 1
            elif len(commit.commit_msg) == 0:
                message_null += 1
            else:
                commit_obj_list_filter.append(commit)
        statics[self.project_name] = (len(commits_list), merge_num, message_null, non_ascill)
        self.save_to_csv(commit_obj_list_filter)

    def save_to_csv(self, commits_list):
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        file_name = os.path.join(SAVE_DIR, str(self.project_name) + '_commits.csv')
        with open(file_name, 'w', ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ['commit_id', 'subject', 'message', 'diff', 'file_changed', 'date', 'author', 'parent_commit', 'parent_number', 'project'])
            for i in range(len(commits_list)):
                commit = commits_list[i]
                try:
                    writer.writerow([commit.hash_id, commit.subject, commit.commit_msg, commit.diff, commit.file_changed,
                                     commit.date, commit.author, commit.parent_commit, commit.parent_commit_number, commit.project_name])
                except:
                    continue


class GitCommit:
    def __init__(self, hash_id, repo, proj_name):
        self.current_commit = repo.commit(hash_id)
        self.hash_id = str(hash_id)
        self.proj_name = proj_name
        self.repo = repo
        self.set_commit_attributes()

    def set_commit_attributes(self):
        try:
            if len(self.current_commit.message) == 0:
                self.commit_msg = ''
            else:
                self.current_commit.message.encode(encoding='utf-8').decode('ascii')
                self.commit_msg = str(self.current_commit.message)
            lines = self.current_commit.message.split('\n')
            self.subject = str(lines[0].strip())
            self.date = str(self.current_commit.authored_datetime)
            self.author = str(self.current_commit.author)
            self.diff = str(self.repo.git.diff(self.parent_commit, self.current_commit))
            self.parent_commit_number = len(self.parent_commit)
            self.file_changed = len(list(self.current_commit.stats.files.keys()))
            self.parent_commit = self.current_commit.parents
            self.parent_commit = self.parent_commit[0].hexsha
        except:
            self.commit_msg = None


def single_process_data_process(project_name, statics):
    project = GitProject(local_project_name=project_name, repo_path=os.path.join(PROJECTS_DIR, project_name))
    project.process_commits(statics)


def multiprocess_projects(project_list, multiprocess=True):
    message_null = 0
    total_num = 0
    non_ascill = 0
    merge_num = 0
    statics = mp.Manager().dict()
    if multiprocess:
        process_list = []
        for project in project_list:
            process = mp.Process(target=single_process_data_process, args=(project, statics))
            process_list.append(process)
            process.start()
        for process in process_list:
            process.join()
    else:
        for project in project_list:
            single_process_data_process(project, statics)
    statics = dict(statics)
    for key in statics.keys():
        message_null += statics[key][2]
        total_num += statics[key][0]
        non_ascill += statics[key][3]
        merge_num += statics[key][1]
    print('total num: %d' % total_num)
    print('null message: %d' % message_null)
    print('non_ascill message: %d' % non_ascill)
    print('merge_num message: %d' % merge_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project_dir', default=PROJECTS_DIR, help="Projects dir need to preprocess")
    args = parser.parse_args()
    # project_list = ['mockito']
    dir_list = os.listdir(args.project_dir)
    project_list = []
    for dir in dir_list:
        if os.path.isdir(os.path.join(args.project_dir, dir)):
            project_list.append(dir)
    print('total projects num %d ' % len(project_list))
    multiprocess_projects(project_list, multiprocess=True)
