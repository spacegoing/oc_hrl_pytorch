# Installation #


- MongoDB: `docker run --name mg -v /home/chli4934/ubCodeLab/volumeb3_data:/data -p 27017:27017 -itd mongo`
- mujoco-py: install check official github repo
- baselines: symlink local repo `/home/chli4934/ubCodeLab/baselines/` to `/home/chli4934/anaconda3/lib/python3.8/site-packages/`

## Install Mujoco ##

- Do Not manually install Nvidia Driver. Use `conda install
  tensorflow-gpu` will automatically install CUDA-toolkit under
  `anaconda3/pkgs`
- Download mujoco and activation key. Add path to `.bashrc`
  ``` bash
  export LD_LIBRARY_PATH="/home/chli4934/.mujoco/mjpro150/bin:$LD_LIBRARY_PATH"
  # add libGLxxxx to path
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
  ```

1. `conda install tensorflow-gpu`
2. conda install pytorch, script from pytorch official website
3. Install `mujocopy` https://github.com/openai/mujoco-py/issues/158
   ``` bash
   github clone https://github.com/openai/mujoco-py.git
   cd mujoco-py
   # switch to 1.50.1.1
   pip install -e .
   
   # after installed, import mujoco it will compile itself
   pip -c 'import mujoco_py'
   ```
4. 

### Dependencies ###

On Ubuntu 18.04

- When import dm_control: `undefined symbol: __glewBlitFramebuffer`
  - https://github.com/deepmind/dm_control/issues/3
  - `sudo apt-get install libglew-dev libglfw3-dev`

### .bashrc ###

```bash
# https://github.com/deepmind/dm_control/issues/97
export LD_LIBRARY_PATH="/home/chli4934/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH"
export MJLIB_PATH=$HOME/.mujoco/mujoco200/bin/libmujoco200.so
export MJKEY_PATH=$HOME/.mujoco/mujoco200/mjkey.txt
export MUJOCO_PY_MJPRO_PATH=$HOME/.mujoco/mujoco200/
export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mujoco200/mjkey.txt
```

# Implementation Ideas #

## Config Options ##

- how to name / record configurations

## Log ##

### Done###
- log value
- log option

## Opt ##

### Gradients ###

- Proof of layer by layer training 
- Not only separate by layer, but also separate by timestep. So
  monthly skill can attend to nanoseconds actions

- ppo clip ratio small -> large
- rollout length short -> long

#### Done ####
- is Qos - Qos_1 correct?
  - use pot*Q_o_st to compute V(Ot|St,Ot-1)
- ppo option adv use vst not qost
  - use o_adv for ppo loss

### Option Nets ###

- better option learning?
- whether use entropy

### Action Nets ###

- action policy pass gradients of option embedding back
  - gradient theorem for this

#### Done ####
- Rather than num_option action nets, use 1 action net with
  different option embedding inputs
- Action net use self attention (transformer encoder)

### Value Functions ###

- action, option have different value functions
- value function use transformer encoding

## Problems ##

### More Independent Between Workers, Better performance ###
- initial run git:1e48131
- Best run, each worker's option independent
- Performance decrease, option choices between workers more
  related
  
#### conclusion ####
- option choice rely too much on previous one
- each period has an advantage option, no matter what state, it
  turns to choose that option

#### Solutions ####
1. decoupling previous option gradients: options used by decoder
   detached from embedding
2. decoupling action option gradients: options used by action do
   not pass gradients back to embedding

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
