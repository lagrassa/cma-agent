"""
Run CMA-ES on an RL task.
"""

from multiprocessing import Pool, cpu_count
import os
import sys
sys.path.append("../baselines")
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
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
                  name="test",
                  expnum=0,

                  log_file=None):
    """
    Run CMA on the environment.
    """
    reward_threshold=-50
    local_variables = learn_setup(env_id=env_id, timesteps=timesteps, param_scale=param_scale, name=name, expnum=expnum, log_file=log_file, n_steps_per_episode=50, reward_threshold=reward_threshold)
    #env = make_vec_env(env_id, "mujoco", 1, None, reward_scale=1.0, flatten_dict_observations=True)
    #if log_file is None:
    #    log_file = os.path.join('results', env_id + '.monitor.csv')
    #env = LoggedEnv(gym.make(env_id), log_file)
    #with tf.Session() as sess:
    #    model = ContinuousMLP(sess, env.action_space, gym_space_vectorizer(env.observation_space))
    #    roller = BasicRoller(env, model, min_episodes=4, min_steps=500)
    #    sess.run(tf.global_variables_initializer())
    #    trainer = CMATrainer(sess, scale=param_scale)

    if True:
        steps = 0
        #rewards = []
        while steps < timesteps:
            #sub_steps, sub_rewards = trainer.train(roller)
            while steps < timesteps:
                sub_steps, _, _ = learn_iter(**local_variables)
                steps += sub_steps
            #rewards.extend(sub_rewards)
            #print('%s: steps=%d mean=%f batch_mean=%f' %
            #      (env_id, steps, np.mean(rewards), np.mean(sub_rewards)))

def learn_iter(roller=None, rewards=None, trainer=None, update=None, env_id=None, reward_threshold=0, success_only=True):
    sub_steps, sub_rewards, infos = trainer.train(roller)
    rewards.extend(sub_rewards)
    #print('%s: steps=%d mean=%f batch_mean=%f' %
    #      (env_id, update, np.mean(rewards), np.mean(sub_rewards)))

    if success_only:
        success_rate = np.mean(np.array(sub_rewards) > reward_threshold)
    else:
        success_rate = np.mean(np.array(sub_rewards))
    return sub_steps, success_rate, infos

def learn_test(roller=None, rewards=None, trainer=None, update=None, env_id=None, n_episodes=None, n_steps_per_iter = None, reward_threshold=0, success_only=True):
    sub_steps, sub_rewards, _ = trainer.train(roller, test = True)
    rewards.extend(sub_rewards)
    #print('%s: steps=%d mean=%f batch_mean=%f' %
    #      (env_id, update, np.mean(rewards), np.mean(sub_rewards)))
    if success_only:
        success_rate = np.mean(np.array(sub_rewards) > reward_threshold)
    else:
        success_rate = np.mean(np.array(sub_rewards))
    return success_rate

def learn_setup(env_id=None,
                  timesteps=int(5e6),
                  env_name = None,
                  param_scale=1,
                  name = "test",
                  expnum=0,
                  env=None,
                  n_episodes = None,
                  n_steps_per_episode=None,
                  reward_threshold=0,
                  CMA_mu=None,
                  CMA_cmean=None,
                  CMA_rankmu=None,
                  CMA_rankone=None,
                  log_file=None):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    if env_id is None:
        env_id = env_name
    if env is None:
        env = make_vec_env(env_id, "mujoco", 1, None, reward_scale=1.0, flatten_dict_observations=True)

    if log_file is None:
        log_file = os.path.join('results', "recent" + name+"_"+str(expnum)+".monitor.csv")
        log_npy = os.path.join('results', "recent"+name+'_'+str(expnum)+'.npy')
    #env = LoggedEnv(env, log_file, log_npy)

    model = ContinuousMLP(sess, env.action_space, gym_space_vectorizer(env.observation_space))
    roller = BasicRoller(env, model, min_episodes=1, min_steps=n_steps_per_episode)
    sess.run(tf.global_variables_initializer())
    trainer = CMATrainer(sess, scale=param_scale, CMA_mu=CMA_mu, CMA_cmean=CMA_cmean, CMA_rankmu=CMA_rankmu, CMA_rankone=CMA_rankone) #, popsize=n_episodes)
    rewards = []
    local_variables = {'roller':roller,
                       'trainer':trainer,
                        'env_id':env_name,
                        'reward_threshold':reward_threshold,
                       'rewards':rewards}
    return local_variables

def run_with_kwargs(kwargs):
    """
    Run an experiment with the kwargs dict.
    """
    training_loop(**kwargs)

if __name__ == '__main__':
    if not os.path.isdir('results'):
        os.mkdir('results')
    run_with_kwargs({'env_id':'FetchReach-v1'})
