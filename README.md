# The Skill-Action Architecture Readme #

Our implementation is based on *DAC: The Double Actor-Critic
Architecture for Learning Options* Shangtong Zhang, Shimon
Whiteson (NeurIPS 2019).
https://github.com/ShangtongZhang/DeepRL/tree/DAC. We deeply
appreciate their efforts on opening source their code. We will
also create a pull request of our code on their branch after
review.

## Installation ##

We include a requirements.txt in this repo. Users may install
this repo by executing

`pip install -r requirements.txt`

However, if anything goes wrong, we suggest readers use original
Shangtong Zhang's requirements.txt. They even provide a
Dockerfile. However, we have not used docker in our implementations.

### Install MongoDB ###

We use MongoDB for storing experiment results. We suggest readers
to use MongoDB Docker: `https://hub.docker.com/_/mongo`.

After docker container is running, users should also have
`pymongo` installed:

`pip install pymongo`

### Install MuJoCo Gym ###

When we install Shangtong Zhang's `requirements.txt`, we hit
several bugs regards to MuJoCo and OpenAI Gym installation.
Basically, users need to apply a license from
http://www.mujoco.org/. If any error with OpenAI Gym / baseline
is hit, users need to refer to their official Github repo. Most
issues are answered there.

## Running SA Experiments ##

To run SA paper's experiments on CPUs, simply run:
`python run_sa.py`

Users may alter `process=1` to number of CPUs you have.

### Hyper-Parameters ###

All hyper-parameters are in `./deep_rl/Params.py`

## Plotting Script ##

All OpenAI MuJoCo experiment figures are plotted using Shangtong
Zhang's original script `plot_mujoco.py` without any parameter
twiking.

Other figures (4,5,6) are all plotted using `analyze_mongo.py`

## Reproduce Our Results ##

Reinforcement learning algorithms are parameters sensitive.
However, we find SA is surprisingly robust. There should be no
problem reproduce our results with any random seed etc. Before
upload this code we run 12 runs of 4 infinite horizon games and
saved log data in `./doe_tf_log`, some of them are actually even
better than we reported in paper. Users can compare their results
with those files as a benchmark.

# Original DAC Readme #

## This branch is the code for the paper ##

*DAC: The Double Actor-Critic Architecture for Learning Options* \
Shangtong Zhang, Shimon Whiteson (NeurIPS 2019)

    .
    ├── Dockerfile                                      # Dependencies
    ├── requirements.txt                                # Dependencies
    ├── template_jobs.py                                
    |   ├── batch_mujoco                                # Start Mujoco experiments 
    |   ├── batch_dm                                    # Start DMControl experiments 
    |   ├── a_squared_c_ppo_continuous                  # Entrance of DAC+PPO
    ├── deep_rl/agent/ASquaredC_PPO_agent.py            # Implementation of DAC+PPO 
    ├── deep_rl/agent/ASquaredC_A2C_agent.py            # Implementation of DAC+A2C 
    ├── deep_rl/agent/AHP_PPO_agent.py                  # Implementation of AHP+PPO 
    ├── deep_rl/agent/IOPG_agent.py                     # Implementation of IOPG 
    ├── deep_rl/agent/OC_agent.py                       # Implementation of OC 
    ├── deep_rl/agent/PPOC_agent.py                     # Implementation of PPOC 
    ├── deep_rl/component/cheetah_backward.py           # Cheetah-Backward 
    ├── deep_rl/component/walker_ex.py                  # Walker-Backward/Squat 
    ├── deep_rl/component/fish_downleft.py              # Fish-Downleft 
    └── plot_paper.py                                   # Plotting

> I can send the data for plotting via email upon request.

> This branch is based on the DeepRL codebase and is left unchanged after I completed the paper. Algorithm implementations not used in the paper may be broken and should never be used. It may take extra effort if you want to rebase/merge the master branch.
