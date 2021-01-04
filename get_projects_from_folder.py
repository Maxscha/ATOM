import os
import time

project_root = os.path.dirname(os.path.dirname(__file__))

RAW_DATA_PATH = os.path.join(project_root, 'ast_process/data/raw_commits_from_github')

list_of_projects = [p[:-12] for p in os.listdir(RAW_DATA_PATH)]

for l in list_of_projects:
    print(l)