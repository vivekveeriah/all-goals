import os.path as osp
import gym
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
from auxiliary_tasks import logger

from mastery.common.misc_util import set_global_seeds
from mastery.common.math_util import explained_variance
from mastery.common.vec_env.subproc_vec_env import SubprocVecEnv
from mastery.common.atari_wrappers import wrap_deepmind

from mastery.utils import discount_with_dones
from mastery.utils import Scheduler, make_path, find_trainable_variables
from mastery.policies import CnnPolicy
from mastery.utils import cat_entropy, mse

class Model(object):
    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        nact_meta = ac_meta_space.n
        nbatch_meta = nenvs * nsteps_meta

        nact = ac_space.n
        nbatch = nenvs * nsteps

        A = tf.placeholder(tf.int32, [nbatch], name="actions")
        R = tf.placeholder(tf.float32, [nbatch], name="rewards")
        LR = tf.placeholder(tf.float32, [], name="step_size")

        A_meta = tf.placeholder(tf.int32, [nbatch_meta], name="meta_actions")
        ADV_meta = tf.placeholder(tf.float32, [nbatch_meta], name="meta_advantage")
        R_meta = tf.placeholder(tf.float32, [nbatch_meta], name="meta_rewards")
        LR_meta = tf.placeholder(tf.float32, [], name="meta_step_size")

        eval_return = tf.placeholder(tf.float32, [], name="evaluated_return")

        step_model = controller_policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        train_model = controller_policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

        step_meta_model = meta_controller_policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        train_meta_model = meta_controller_policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_meta_model.pi, labels=A_meta)
        pg_loss = tf.reduce_mean(ADV_meta * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_meta_model.vf), R_meta))
        entropy = tf.reduce_mean(cat_entropy(train_meta_model.pi))
        meta_controller_loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        tf.summary.scalar('meta_model/neglogpac', tf.reduce_mean(neglogpac))
        tf.summary.scalar('meta_model/pg_loss', tf.reduce_mean(pg_loss))
        tf.summary.scalar('meta_model/vf_loss', tf.reduce_mean(vf_loss))
        tf.summary.scalar('meta_model/entropy_term', tf.reduce_mean(entropy))
        tf.summary.scalar('meta_model/meta_controller_loss', tf.reduce_mean(meta_controller_loss))

        tf.summary.scalar('meta_model/actions', tf.reduce_mean(A_meta))
        tf.summary.scalar('meta_model/advantage', tf.reduce_mean(ADV_meta))
        tf.summary.scalar('meta_model/rewards', tf.reduce_mean(R_meta))
        tf.summary.scalar('meta_model/step_size', tf.reduce_mean(LR_meta))

        tf.summary.scalar('model/actions', tf.reduce_mean(A))
        tf.summary.scalar('model/rewards', tf.reduce_mean(R))
        tf.summary.scalar('model/step_size', tf.reduce_mean(LR))

        # add the loss variable summaries for controller

        tf.summary.scalar('model/eval_return', tf.reduce_mean(eval_return))

        controller_params = find_trainable_variables('controller')
        controller_grads = tf.gradients(controller_loss, controller_params)
        for var in controller_params:
            tf.summary.histogram(var.name, var)
        grads_with_controller_vars = list(zip(controller_grads, controller_params))
        for grad, var in grads_with_controller_vars:
            tf.summary.histogram(var.name + '/grad', grad)

        if max_grad_norm_controller is not None:
            controller_grads, controller_grads_norm = tf.clip_by_global_norm(controller_grads, max_grad_norm_controller)

        clipped_grads_with_controller_vars = list(zip(controller_grads, controller_params))
        for grad, var in clipped_grads_with_controller_vars:
            tf.summary.histogram(var.name + '/clipped_grad', grad)

        meta_controller_params = find_trainable_variables('meta_controller')
        meta_controller_grads = tf.gradients(meta_controller_loss, meta_controller_params)
        for var in meta_controller_params:
            tf.summary.histogram(var.name, var)



class Runner(object):

    def __init__(self, env, model, eval_env=None, nsteps=5, nstack=4, gamma=0.99):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv * nsteps, nh, nw, nc * nstack)
        self.obs = np.zeros((nenv, nh, nw, nc*nstack), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

        self.eval_env = eval_env
        self.eval_obs = np.zeros((1, nh, nw, nc * nstack), dtype=np.uint8)
        eval_obs = self.eval_env.reset()
        self.update_eval_obs(eval_obs)
        self.eval_initial_state = model.initial_state
        self.eval_state = model.initial_state
        self.eval_dones = [False for _ in range(1)]

        # if eval_env is not None:
        #     self.eval_env = eval_env
        #     self.eval_obs = np.zeros((1, nh, nw, nc * nstack), dtype=np.uint8)
        #     eval_obs = eval_env.reset()
        #     self.update_eval_obs(eval_obs)
        #     self.eval_state = model.initial_eval_state

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs = np.roll(self.obs, shift=-self.nc, axis=3)
        self.obs[:, :, :, -self.nc:] = obs

    def update_eval_obs(self, eval_obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.eval_obs = np.roll(self.eval_obs, shift=-self.nc, axis=3)
        self.eval_obs[:, :, :, -self.nc:] = eval_obs

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.update_obs(obs)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
            #
            # if n == 0 and True in dones:
            #     print(dones)
            #     print(rewards)

        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values

    def eval(self, eval_steps):
        self.eval_obs = np.zeros(self.eval_obs.shape, dtype=np.uint8)
        eval_obs = self.eval_env.reset()
        self.update_eval_obs(eval_obs)
        self.eval_state = self.eval_initial_state
        self.eval_dones = [False for _ in range(1)]

        G = 0.0

        for _ in range(eval_steps):
            actions, _, states = self.model.step(self.eval_obs, self.eval_state, self.eval_dones)
            obs, rewards, dones, _ = self.eval_env.step(actions)
            self.update_eval_obs(obs)
            self.eval_state = states
            self.eval_dones = dones
            G += rewards[0]
            for n, done in enumerate(dones):
                if done:
                    return G
        return G

def learn(policy, env, seed, eval_env=None, eval_interval=None, eval_steps=None, nsteps=5, nstack=4, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes) # HACK
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
            max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    runner = Runner(env, model, eval_env=eval_env, nsteps=nsteps, nstack=nstack, gamma=gamma)

    nbatch = nenvs * nsteps
    tstart = time.time()
    evaluated_return = -1000

    for update in range(1, total_timesteps // nbatch + 1):
        obs, states, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values, update, evaluated_return)
        nseconds = time.time() - tstart
        fps = int((update*nbatch)/nseconds)

        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("eval_return", float(evaluated_return))
            logger.dump_tabular()

        if update % eval_interval == 0:
            evaluated_return = runner.eval(eval_steps)

    env.close()