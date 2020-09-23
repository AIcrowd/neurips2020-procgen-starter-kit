#!/usr/bin/env python

import yaml
import subprocess
import sys


def get_experiment_config_file():
    exp = subprocess.run(
        'bash -c "source ./run.sh > /dev/null; echo \$EXPERIMENT"',
        shell=True,
        capture_output=True
    )
    config_file = exp.stdout.decode().strip()
    print("Using config from", config_file)
    return config_file


def read_yaml(fpath):
    with open(fpath) as fp:
        data = yaml.safe_load(fp)
    return list(data.values())[0]


def get_num_workers(ctx):
    num_workers = ctx.get("config", {}).get("num_workers", 0)
    disable_evaluation_worker = ctx.get("disable_evaluation_worker", False)
    if not disable_evaluation_worker:
        num_workers += 1
    return num_workers


def check_cpu_allotment(ctx):
    assert get_num_workers(ctx) < 8, "Please make sure that `num_workers + 1 <= 8`"


def check_gpu_allotment(ctx):
    num_workers = get_num_workers(ctx)
    num_gpus = ctx.get("config", {}).get("num_gpus", 0)
    num_gpus_per_worker = ctx.get("config", {}).get("num_gpus_per_worker", 0)
    if num_gpus == 0 or num_gpus_per_worker == 0:
        print("WARNING: You are not using GPUs. Please set `num_gpus` and `num_gpus_per_worker` to a non zero value.")
    assert num_gpus + (num_workers * num_gpus_per_worker) <= 1,\
        "Please make sure that `num_gpus + num_gpus_per_worker * (num_workers+1) <= 8`"


def main():
    config_path = get_experiment_config_file()
    ctx = read_yaml(config_path)
    check_cpu_allotment(ctx)
    check_gpu_allotment(ctx)


if __name__ == '__main__':
    try:
        main()
    except AssertionError as e:
        print(e)
        sys.exit(1)