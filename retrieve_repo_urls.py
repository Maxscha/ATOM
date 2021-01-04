import requests
import json

import os
import time

project_root = os.path.dirname(os.path.dirname(__file__))

RAW_DATA_PATH = os.path.join(project_root, 'ast_process/data/raw_commits_from_github')

list_of_projects = [p[:-12] for p in os.listdir(RAW_DATA_PATH)]

project = 'activemq'


def get_url(project):
    url = f"https://api.github.com/search/repositories?q={project}"
    payload={}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    parsed_response = json.loads(response.text)
    return parsed_response["items"][0]["git_url"]

for p in list_of_projects:
    print(get_url(p))
    time.sleep(60/10) #Rate limit of github api 10 request per minute