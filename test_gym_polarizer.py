import gym_polarizer # Register the environment 'polarizer-v0'
import numpy as np
from matplotlib import pyplot as plt
import gym
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import os


#save_dir = '/tmp/tensorboard/'
save_dir = '/home/anm16/tensorboard/'
parameter_save_path = os.path.join(save_dir, 'polarizer_save_data')

# Defined in https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/ppo2/ppo2.py
    # Proximal Policy Optimization algorithm (GPU version).
    # Paper: https://arxiv.org/abs/1707.06347
    # :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    # :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    # :param gamma: (float) Discount factor
    # :param n_steps: (int) The number of steps to run for each environment per update
    #     (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    # :param ent_coef: (float) Entropy coefficient for the loss caculation
    # :param learning_rate: (float or callable) The learning rate, it can be a function
    # :param vf_coef: (float) Value function coefficient for the loss calculation
    # :param max_grad_norm: (float) The maximum value for the gradient clipping
    # :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    # :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
    #     the number of environments run in parallel should be a multiple of nminibatches.
    # :param noptepochs: (int) Number of epoch when optimizing the surrogate
    # :param cliprange: (float or callable) Clipping parameter, it can be a function
    # :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    # :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    # :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    

# Custom policy - from https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html
class CustomMlpLnLstmPolicy(MlpLnLstmPolicy):
    def __init__(self, *args, **kwargs):
        super(MlpLnLstmPolicy, self).__init__(*args, **kwargs,
                     n_lstm=32, # Default 256
                     layers = [32,32], # Default [64,64]
                     feature_extraction = 'mlp',
                     )
 


def learning_rate_anneal(proportion):
    """ The learning rate can be a function accepting a proportion-of-steps-
    completed value from 1 (time=0) to 0 (time=end), so this one performs annealing """
    start = 1e-2
    stop = 1e-5
    learning_rate = np.exp((np.log(stop)-np.log(start))*(1-proportion) + np.log(start))
    return learning_rate

env = gym.make('polarizer-v0')
# env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run


# Under-development MLP + LSTM policy
model = PPO2(MlpLstmPolicy, env,
             verbose=1,
             nminibatches = 1,
             learning_rate=learning_rate_anneal,
             tensorboard_log=save_dir,
#             noptepochs=1, # Needed until update_fac bug is fixed
             )

model.learn(total_timesteps=int(1e5))

## Working MLP implementation
#model = PPO2(MlpPolicy, env, verbose=1,
#                gamma=0.99, n_steps=2048, ent_coef=0.01, learning_rate=learning_rate_anneal, vf_coef=0.5,
#                max_grad_norm=0.5, lam=0.95, nminibatches=8, noptepochs=8, cliprange=0.2,
#                tensorboard_log=save_dir, _init_setup_model=True)
#model.learn(total_timesteps=int(30e6))


model.save(parameter_save_path)
#%%
#model.load(parameter_save_path)

all_obs = []
all_rewards = []
all_dones = []

obs = env.reset()
for i in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    all_obs.append(obs[0])
    all_rewards.append(rewards[0])
    all_dones.append(dones[0])
    env.render()

all_obs = np.array(all_obs)
all_rewards = np.array(all_rewards)
fig, axs = plt.subplots(3,1, sharex = True)
axs[0].plot(all_rewards)
axs[1].plot(all_obs)
axs[2].plot(all_dones)
plt.show()
