import json
from common import dict_to_features, categories


def any_path_startswith(pattern, files_changed):
    return any(path.startswith(pattern) for path in files_changed)


def autocategorize_from_features(commit_hash, features):
    return autocategorize(commit_hash, *features)


def autocategorize(commit_hash, title, body, pr_number, files_changed, labels):
    if pr_number is None:
        return 'skip'

    if any_path_startswith('torch/onnx', files_changed):
        return 'onnx'
    if 'module: onnx' in labels:
        return 'onnx'
    if '[onnx]' in title.lower():
        return 'onnx'

    if any_path_startswith('torch/csrc/distributed/rpc', files_changed):
        return 'rpc'
    if any_path_startswith('torch/distributed/autograd', files_changed):
        return 'rpc'
    if any_path_startswith('torch/distributed/optim', files_changed):
        return 'rpc'
    if any_path_startswith('torch/distributed/rpc', files_changed):
        return 'rpc'
    if 'module: rpc' in labels:
        return 'rpc'
    if any('/rpc' in path for path in files_changed):
        return 'rpc'

    if any_path_startswith('torch/distributed', files_changed):
        return 'distributed'
    if any_path_startswith('torch/csrc/distributed', files_changed):
        return 'distributed'
    if any('c10d' in path for path in files_changed):
        return 'distributed'
    if 'module: distributed' in labels:
        return 'distributed'

    if 'quantization' in labels:
        return 'quantization'
    if '[quant]' in title.lower():
        return 'quantization'
    if any_path_startswith('torch/quantization', files_changed):
        return 'quantization'
    if any('quantization' in path for path in files_changed):
        return 'quantization'
    if any('quantized' in path for path in files_changed):
        return 'quantization'

    if 'module: distributions' in labels:
        return 'distributions'
    if any('distributions' in path for path in files_changed):
        return 'distributions'

    if any_path_startswith('torch/csrc/jit/mobile/', files_changed):
        return 'mobile'
    if any_path_startswith('android/', files_changed):
        return 'mobile'
    if any_path_startswith('ios/', files_changed):
        return 'mobile'
    if 'mobile' in title.lower():
        return 'mobile'

    if '[caffe2]' in title.lower():
        return 'caffe2'
    if any_path_startswith('caffe2/', files_changed):
        return 'caffe2'

    if any('tensorboard' in path for path in files_changed):
        return 'visualization'
    if 'tensorboard' in title.lower():
        return 'visualization'

    if 'rocm' in title.lower():
        return 'amd'
    if 'hip' in title.lower():
        return 'amd'
    if 'miopen' in title.lower():
        return 'amd'
    if 'module: rocm' in labels:
        return 'amd'

    # cpp api
    if any_path_startswith('test/cpp_api_parity', files_changed):
        return 'cpp'
    if any_path_startswith('test/cpp/api', files_changed):
        return 'cpp'
    if any_path_startswith('torch/csrc/api', files_changed):
        return 'cpp'
    if 'c++ api' in title.lower():
        return 'cpp'
    if 'libtorch' in title.lower():
        return 'cpp'

    if '[jit]' in title.lower():
        return 'jit'
    if 'torchscript' in title.lower():
        return 'jit'
    if 'jit' in labels:
        return 'jit'
    if any_path_startswith('test/test_jit', files_changed):
        return 'jit'

    # python or misc?
    return 'python'


def main():
    with open('data.json') as f:
        data = json.load(f)
    flat_data = [[commit, dict_to_features(data_dict)]
                 for commit, data_dict in data.items()]
    commits, features = zip(*flat_data)
    labels = [autocategorize(commit, *feature)
              for commit, feature in zip(commits, features)]
    for category in categories:
        sharded = [[commit, feature] for commit, feature, label in zip(commits, features, labels)
                   if label == category]
        with open(f'results/{category}.txt', 'w') as f:
            for commit, feature in sharded:
                f.write(f'{commit}  # {feature.title}\n')
                # f.write(f'{feature.title} [#{feature.pr_number}](https://github.com/pytorch/pytorch/pull/{feature.pr_number})\n')

if __name__ == '__main__':
    main()
