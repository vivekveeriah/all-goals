import random
import numpy as np
import os
import pickle


class ReplayBuffer():
    def __init__(self, memory_size):
        self._storage = []
        self._maxsize = memory_size
        self._next_idx = 0

    def lengthBuffer(self):
        return len(self._storage)

    def add(self, obs_t, a_t, r_t, obs_tp1, done):
        data = (obs_t, a_t, r_t, obs_tp1, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size, goal_obses):
        experience_batch = random.sample(self._storage, batch_size)
        return self._encodeSamples(experience_batch, goal_obses)

    def _encodeSamples(self, experience_batch, goal_obses):
        batch_size = len(experience_batch)
        num_goals = len(goal_obses)
        h, w, c = self._storage[0][0].shape
        obs_ret = np.zeros((batch_size * num_goals, h, w, c))
        act_ret = np.zeros((batch_size * num_goals, ))
        rew_ret = np.zeros((batch_size * num_goals, ))
        next_obs_ret = np.zeros((batch_size * num_goals, h, w, c))
        done_ret = np.zeros((batch_size * num_goals, ))
        goal_ret = np.zeros((batch_size * num_goals, h, w, c))

        for i in range(batch_size):
            obs, act, rew, next_obs, done = experience_batch[i]
            for goal_idx in range(num_goals):
                pos = (goal_idx * batch_size) + i
                obs_ret[pos] = obs
                act_ret[pos] = act
                next_obs_ret[pos] = next_obs
                goal_ret[pos] = goal_obses[goal_idx]
                if np.array_equal(goal_obses[goal_idx], obs):
                    rew_ret[pos] = 0.
                    done_ret[pos] = True
                else:
                    rew_ret[pos] = -0.1
                    done_ret[pos] = False

        return obs_ret, act_ret, rew_ret, next_obs_ret, done_ret, goal_ret

    def saveReplayBuffer(self):
        if not os.path.isdir('replay_buffer_saved'):
            os.makedirs('replay_buffer_saved')
        pickle.dump(self._storage, open('replay_buffer_saved/replay_buffer.pkl', 'wb'))
