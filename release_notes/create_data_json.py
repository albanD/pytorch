import requests
import subprocess
import sys
import locale
import re
import json
from common import get_features
from functools import partial


def readlines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if len(line) > 0]
        return lines


def main():
    commits = readlines('commit_list.txt')

    features = map(partial(get_features, return_dict=True), commits)
    data = {commit: feature for commit, feature in zip(commits, features)
            if feature is not None}
    with open('data.json', 'w') as outfile:
        json.dump(data, outfile)


main()
