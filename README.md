# NeurIPS 2020 - Procgen Starter Kit
![AIcrowd-Logo](https://raw.githubusercontent.com/AIcrowd/AIcrowd/master/app/assets/images/misc/aicrowd-horizontal.png)

This is the starter kit for the [NeurIPS 2020 - Procgen competition](https://www.aicrowd.com/challenges/neurips-2020-procgen-competition) hosted on [AIcrowd](https://www.aicrowd.com/)

# 🕵️ About ProcGen Benchmark

16 simple-to-use procedurally-generated [gym](https://github.com/openai/gym) environments which provide a direct measure of how quickly a reinforcement learning agent learns generalizable skills.  The environments run at high speed (thousands of steps per second) on a single core.

![](https://raw.githubusercontent.com/openai/procgen/master/screenshots/procgen.gif)

These environments are associated with the paper [Leveraging Procedural Generation to Benchmark Reinforcement Learning](https://cdn.openai.com/procgen.pdf) [(citation)](#citation). Compared to [Gym Retro](https://github.com/openai/retro), these environments are:

* Faster: Gym Retro environments are already fast, but Procgen environments can run >4x faster.
* Non-deterministic: Gym Retro environments are always the same, so you can memorize a sequence of actions that will get the highest reward.  Procgen environments are randomized so this is not possible.
* Customizable: If you install from source, you can perform experiments where you change the environments, or build your own environments.  The environment-specific code for each environment is often less than 300 lines.  This is almost impossible with Gym Retro.

# 🔧 Installation

## Quick start

```bash
git clone https://github.com/AIcrowd/neurips2020-procgen-starter-kit.git
./utils/setup.sh
```

## Manually setup the environment

- Clone this repository

```bash
git clone https://github.com/AIcrowd/neurips2020-procgen-starter-kit.git
```

- Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (if you don't have it already)

- Create a new environment with python 3.7

```bash
conda create -n procgen -y
conda activate procgen
conda install python=3.7 -y
```

- Install dependencies

```bash
conda activate procgen
pip install ray[rllib]==0.8.6
pip install procgen
```

- Install the DL framework of you choice (tensorflow, pytorch, etc.,)

# 💪 Getting Started

### Train your agent

```
./run.sh --train
```

### Rollout the agent

```
./run.sh --rollout
```

Please refer to the instructions [here](https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/master/docs/running.md) for advanced users.

## How do I add a custom Model ?
To add a custom model, create a file inside `models/` directory and name it `models/my_vision_network.py`.
Please refer [here](https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/master/models/my_vision_network.py ) for a working implementation of how to add a custom model. You can then set the `custom_model` field in the experiment yaml to `my_vision_network` to cause that model to be used.

## How do I add a custom Algorithm/Trainable/Agent ?
Please refer to the instructions [here](https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/master/experiments/procgen-starter-example.yaml#L4)

## What configs parameters do I have available ?
This is a very thin wrapper over RLLib. Hence, you have all the RLLib config options available in the experiment configuration (YAML) file to play with.
Please refer to [this example](https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/master/experiments/procgen-starter-example.yaml) for an explanation on the structure and available options in the experiment config file.

# 🚀 Submission

## Repository Structure

```
.
├── aicrowd_helpers                     # Helper files (DO NOT EDIT)
├── algorithms                          # Directory to implement your custom algorithm/trainable/agent
│   ├── custom_random_agent
│   ├── random_policy
│   ├── __init__.py
│   └── registry.py                     # Register your custom agents here
├── envs
│   ├── __init__.py
│   ├── framestack.py                   # Example for using custom env wrappers
│   ├── procgen_env_wrapper.py          # Base env used during evaluations (DO NOT EDIT)
├── experiments                         # Directory contaning the config for different experiments
│   ├── impala-baseline.yaml            # Baseline using impala
│   ├── procgen-starter-example.yaml    # Sample experiment config file
│   └── random-policy.yaml              # Sample random policy config file
├── models                              # Directory to implement custom models
│   ├── impala_cnn_tf.py
│   ├── impala_cnn_torch.py
│   └── my_vision_network.py
├── preprocessors                       # Directory to implement your custom observation wrappers
│   ├── __init__.py                     # Register your preprocessors here
│   └── custom_preprocessor.py
├── utils                               # Helper scripts for the competition
│   ├── setup.sh                        # Setup local procgen environment using `conda`
│   ├── submit.sh                       # Submit your solution
│   ├── teardown.sh                     # Remove the existing local procgen environment using `conda`
│   ├── validate_config.py              # Validate the experiment YAML file
│   └── loader.py
├── Dockerfile                          # Docker config for your submission environment
├── aicrowd.json                        # Submission config file (required)
├── callbacks.py                        # Custom Callbacks & Custom Metrics
├── requirements.txt                    # These python packages will be installed using `pip`
├── rollout.py                          # Rollout script (DO NOT EDIT)
├── run.sh                              # Entrypoint to your submission
└── train.py                            # Script to trigger the training using `rllib` (DO NOT EDIT)

```

## `aicrowd.json`
Your repository should have an aicrowd.json file with following fields:

```json
{
    "challenge_id" : "evaluations-api-neurips-2020-procgen",
    "grader_id": "evaluations-api-neurips-2020-procgen",
    "authors" : ["aicrowd-bot"],
    "description" : "NeurIPS 2020: Procgen Competition Agent",
    "docker_build" : false
}
```

This file is used to identify your submission as a part of the NeurIPS 2020 Procgen Competition Challenge. You must use the `challenge_id`, and `grader_id` as specified above.

## Submission environment configuration
By default we will run your code in an Ubuntu 18.04 environment with `tensorflow==1.14.0`, `ray[rllib]==0.8.5` and `procgen` installed.

If you want to run your submissions in a customized environment, first head to `aicrowd.json` and set `docker_build` to `true`. This flag tells that you need a custom environment.

You can specify your software environment by using `Dockerfile``requirements.txt`. Available options are

- `requirements.txt`: We will use `pip install -r requiremens.txt` to install your packages.
- `Dockerfile`: We will build the docker image using the specified Dockerfile. **If you have a Dockerfile in your repository, any other automatic installation will not be triggered.** This means that you will have to include installation steps for packages in `requirements.txt` yourself.

A sample [`Dockerfile`](Dockerfile) and a corresponding [`requirements.txt`](requirements.txt) are provided in this repository for you reference.


## Code entrypoint
The evaluator will read the value of `EXPERIMENT` from `run.sh` in the repository. During the training phase, we will use the file provided by `EXPERIMENT` variable to launch the training. During the rollouts phase, we will use the most recent checkpoint and start rollouts with that.

## Submitting

### Initial setup

1) Setup an AIcrowd GitLab account if you don't have one by going to https://gitlab.aicrowd.com/
2) [Add your SSH key](https://discourse.aicrowd.com/t/how-to-add-ssh-key-to-gitlab/2603)
3) If you accept the challenge rules, click the `Participate` button on [the AIcrowd contest page](https://www.aicrowd.com/challenges/neurips-2020-procgen-competition)

```bash
# Optional: Start working with starter kit as starting point.
git clone git@github.com:AIcrowd/neurips2020-procgen-starter-kit.git
cd neurips2020-procgen-starter-kit

# Add AICrowd git remote endpoint
git remote add aicrowd git@gitlab.aicrowd.com:<your-aicrowd-username>/neurips-2020-procgen-starter-kit.git
git push aicrowd master
```

### Submitting using the helper script
```bash
./utils/submit.sh "<your submission message>"
```

For example,
```bash
./utils/submit.sh "impala-ppo-v0.1"
```

### Manually submitting the code
Create a submission by pushing a tag to your repository with a prefix `submission-`. An example is shown below (you can keep repository name as you desire or have multiple repositories):

```bash
# Create a tag for your submission and push
git tag -am "submission-v0.1" submission-v0.1
git push aicrowd submission-v0.1

# Note : If the contents of your repository (latest commit hash) does not change, 
# then pushing a new tag will not trigger a new evaluation.
```

5) You should now be able to see the details of your submission in the repository's issues page.
`https://gitlab.aicrowd.com/<your-aicrowd-username>/neurips-2020-procgen-starter-kit/issues`

and something along the lines of : 
![](https://i.imgur.com/0dMFZ9x.png)

Happy Submitting!! :rocket:


# Author(s)
- [Sharada Mohanty](https://twitter.com/MeMohanty/)
- [Karl Cobbe](https://github.com/kcobbe)
- [Jyotish](https://github.com/jyotishp)
- [Shivam Khandelwal](https://github.com/skbly7)
