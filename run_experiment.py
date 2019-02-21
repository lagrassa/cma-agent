"""
Run CMA-ES on an RL task.
"""

from multiprocessing import Pool, cpu_count
import os

from anyrl.envs.wrappers.logs import LoggedEnv
from anyrl.rollouts import BasicRoller
from anyrl.spaces import gym_space_vectorizer
import gym
import numpy as np
import tensorflow as tf

from cma_agent import CMATrainer, ContinuousMLP

# pylint: disable=R0913
def training_loop(env_id=None,
                  timesteps=int(5e6),
                  param_scale=1,
                  name = "test",
                  expnum=0,
                  log_file=None):
    """
    Run CMA on the environment.
    """
    local_variables = learn_setup(env_id=env_id, timesteps=timesteps, param_scale=param_scale, name=name, expnum=expnum, log_file=log_file)
    steps = 0
    while steps < timesteps:
        sub_steps, success_rates = learn_iter(**local_variables)
        steps += sub_steps

def learn_iter(roller=None, rewards=None, trainer=None):
    sub_steps, sub_rewards = trainer.train(roller)
    rewards.extend(sub_rewards)
    print('%s: steps=%d mean=%f batch_mean=%f' %
          (env_id, steps, np.mean(rewards), np.mean(sub_rewards)))
    success_rate = None
    return sub_steps, success_rate

def learn_setup(env_id=None, 
                  timesteps=int(5e6),
                  param_scale=1,
                  name = "test",
                  expnum=0,
                  log_file=None):
    sess = tf.Session()
    if log_file is None:
        log_file = os.path.join('results', env_id + name+"_"+str(expnum)+".monitor.csv")
        log_npy = os.path.join('results', env_id+name+'_'+str(expnum)+'.npy')
    env = LoggedEnv(gym.make(env_id), log_file, log_npy)
    model = ContinuousMLP(sess, env.action_space, gym_space_vectorizer(env.observation_space))
    roller = BasicRoller(env, model, min_episodes=3, min_steps=30)
    sess.run(tf.global_variables_initializer())
    trainer = CMATrainer(sess, scale=param_scale)
    rewards = []
    local_variables = {'roller':roller,
                       'trainer':trainer,
                       'rewards':reward}
    return local_variables


def run_with_kwargs(kwargs):
    """
    Run an experiment with the kwargs dict.
    """
    training_loop(**kwargs)

if __name__ == '__main__':
    import sys
    if not os.path.isdir('results'):
        os.mkdir('results')
    run_with_kwargs({'env_id':'Pendulum-v0', 'param_scale':1.5, 'expnum':int(sys.argv[2]), 'name':sys.argv[1]})

