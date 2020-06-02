# Benchmarking tool for the autograd API

The main script can be run with `python functional_autograd_benchmark.py --help`.

The main usage for this script is to run it before your change (using as output `before.txt`) and after your change (using as output `after.txt`).
You can then use `python compare.py` to get a markdown table comparing the two runs that can be posted in your PR.
