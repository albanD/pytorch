import argparse
import json
from common import dict_to_features, get_features
import re

def main(path):
    path_pattern = r'results/(.*)\.txt'
    matches = re.findall(path_pattern, path)
    assert len(matches) == 1
    category = matches[0]

    with open('data.json') as f:
        cache = json.load(f)
    with open(path, 'r') as f:
        commits = f.readlines()
    commits = [commit.split('#')[0].strip() for commit in commits]
    outpath = f'pretty_results/{category}.md'

    with open(outpath, 'w+') as f:
        for commit in commits:
            if commit in cache.keys():
                features = dict_to_features(cache[commit])
            else:
                features = get_features(commit)
            num = features.pr_number
            text = f'* {features.title} ([#{num}](https://github.com/pytorch/pytorch/pull/{num})).\n'
            f.write(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert list of commits to markdown')
    parser.add_argument('file', help='which file to convert. Must look like results/{category}.txt')
    args = parser.parse_args()
    main(args.file)
