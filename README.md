# Implementation Ideas #

## Discrepancy ##

I altered
`/home/chli4934/ubCodeLab/baselines/baselines/common/vec_env/vec_env.py`
to `vec_env.py.episode_step` by adding `episodic_flag` into
`step_wait()`:

```python
class VecEnv(ABC):
    def step(self, actions, episodic_flag=True):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait(episodic_flag)
```

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
