import numpy as np
import tensorflow as tf
from mastery.utils import conv, fc, fc_embedding, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample

class LnLstmPolicy(object):
    def __init__(self, sess, obs_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = obs_space.shape
        obs_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, obs_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states
        with tf.variable_scope('model', reuse=reuse):
            h = conv(tf.cast(X, tf.float32) / 255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x: x)
            vf = fc(h5, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(obs, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X: obs, S: state, M: mask})
            return a, v, s

        def value(obs, state, mask):
            return sess.run(v0, {X: obs, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):
    def __init__(self, sess, obs_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = obs_space.shape
        obs_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, obs_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states
        with tf.variable_scope('model', reuse=reuse):
            h = conv(tf.cast(X, tf.float32) / 255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x: x)
            vf = fc(h5, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(obs, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X: obs, S: state, M: mask})
            return a, v, s

        def value(obs, state, mask):
            return sess.run(v0, {X: obs, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):
    def __init__(self, sess, obs_space, ac_space, nenv, nsteps, nstack, is_eval=False, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = obs_space.shape
        obs_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space.n
        # X = tf.placeholder(tf.uint8, shape=obs_shape)  # obs
        X = tf.placeholder(tf.uint8, shape=(None, nh, nw, nc * nstack))
        with tf.variable_scope('model', reuse=reuse):
            h = conv(tf.cast(X, tf.float32) / 255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x: x)
            vf = fc(h4, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = []  # not stateful

        def step(obs, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X: obs})
            return a, v, []  # dummy state

        def value(obs, *_args, **_kwargs):
            return sess.run(v0, {X: obs})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class MasteryPolicy(object):
    def __init__(self, sess, obs_space, ac_space, nenv, nsteps, nstack=1, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = obs_space.space
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, shape=(None, nh, nw, nc * nstack))
        GX = tf.placeholder(tf.uint8, shape=(None, nh, nw, nc * nstack))
        with tf.variable_scope('model', reuse=reuse):
            # check the filter sizes and strides for conv layers
            h = conv(tf.cast(X, tf.float32) / 255., 'c1', nf=16, rf=2, stride=1, init_scale=np.sqrt(2))
            h1 = conv(h, 'c2', nf=32, rf=2, stride=1, init_scale=np.sqrt(2))
            h1 = conv_to_fc(h1)
            h2 = fc(h1, 'fc1', nh=512, init_scale=np.sqrt(2))
            h3 = fc_embedding(h2, 'x_emb', nh=1024, act=lambda x: x)

            hg = conv(tf.cast(GX, tf.float32) / 255., 'c1', nf=16, rf=2, stride=1, init_scale=np.sqrt(2))
            hg1 = conv(hg, 'c2', nf=32, rf=2, stride=1, init_scale=np.sqrt(2))
            hg1 = conv_to_fc(hg1)
            gh2 = fc(hg1, 'fc1', nh=512, init_scale=np.sqrt(2))
            gh3 = fc_embedding(gh2, 'g_emb', nh=1024, act=lambda x: x)

            h4 = tf.multiply(h3, gh3)
            h5 = fc(h4, 'fc2', nh=512, init_scale=np.sqrt(2))
            h6 = fc(h5, 'fc3', nh=256, init_scale=np.sqrt(2))
            q = fc(h6, 'q', nact, act=lambda x: x)

        q0 = q
        self.initial_state = []

        def step(obs, goal_obs, *_args, **_kwargs):
            q_value = sess.run([q0], {X: obs, GX: goal_obs})
            return q_value, []

        def value(obs, goal_obs, *_args, **_kwargs):
            q_value = sess.run([q0], {X: obs, GX: goal_obs})
            return q_value, []

        self.X = X
        self.GX = GX
        self.q = q0
        self.step = step
        self.value = value

class MetaControllerPolicy(object):
    def __init__(self, sess, obs_space, ac_space, nenv, nsteps, nstack=1, nlstm=256, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = obs_space.shape
        obs_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.float32, obs_shape)
        M = tf.placeholder(tf.float32, [nbatch])
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2])
        with tf.variable_scope('meta_model', reuse=reuse):
            h = conv(tf.cast(X, tf.float32) / 255., 'c1', nf=16, rf=2, stride=1, init_scale=np.sqrt(2))
            h1 = conv(h, 'c2', nf=32, rf=2, stride=1, init_scale=np.sqrt(2))
            h1 = conv_to_fc(h1)

            # concat other input variables to h1 here

            h2 = fc(h1, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h2, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h3, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h3 = seq_to_batch(h3)
            pi = fc(h3, 'pi', nact, act=lambda x: x)
            vf = fc(h3, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(obs, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X: obs, S: state, M: mask})
            return a, v, s

        def value(obs, state, mask):
            return sess.run(v0, {X: obs, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
