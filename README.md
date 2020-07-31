# Implementation Ideas #

## Config Options ##

- how to name / record configurations

## Log ##

- log value
- log option

## Opt ##

### Gradients ###

- is Qos - Qos_1 correct?
- ppo clip ratio small -> large
- ppo option adv use vst not qost

### Option Nets ###

- better option learning?
- whether use entropy

### Action Nets ###

- Action net use self attention (transformer encoder)
- Rather than num_option action nets, use 1 action net with
  different option embedding inputs

### Value Functions ###

- action, option have different value functions
- value function use transformer encoding


# Origin Readme #

This branch is the code for the paper

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
