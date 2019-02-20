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
    if log_file is None:
        log_file = os.path.join('results', env_id + name+"_"+str(expnum)+".monitor.csv")
        log_npy = os.path.join('results', env_id+name+'_'+str(expnum)+'.npy')
    env = LoggedEnv(gym.make(env_id), log_file, log_npy)
    with tf.Session() as sess:
        model = ContinuousMLP(sess, env.action_space, gym_space_vectorizer(env.observation_space))
        roller = BasicRoller(env, model, min_episodes=3, min_steps=30)
        sess.run(tf.global_variables_initializer())
        trainer = CMATrainer(sess, scale=param_scale)
        steps = 0
        rewards = []
        while steps < timesteps:
            sub_steps, sub_rewards = trainer.train(roller)
            steps += sub_steps
            rewards.extend(sub_rewards)
            print('%s: steps=%d mean=%f batch_mean=%f' %
                  (env_id, steps, np.mean(rewards), np.mean(sub_rewards)))

def run_with_kwargs(kwargs):
    """
    Run an experiment with the kwargs dict.
    """
    training_loop(**kwargs)

if __name__ == '__main__':
    import sys
    if not os.path.isdir('results'):
        os.mkdir('results')
    run_with_kwargs({'env_id':'StirEnv-v0', 'param_scale':1.5, 'expnum':int(sys.argv[2]), 'name':sys.argv[1]})

