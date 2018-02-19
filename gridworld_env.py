import os
import sys
import numpy as np
import pygame
from scipy.misc import imresize
import scipy.misc
import gym

class ActionSpace():
    def __init__(self, num_actions):
        self.n = num_actions

class ObservationSpace():
    def __init__(self, obs_shape):
        self.shape = obs_shape

class GridWorld():
    str_MDP = ''
    num_rows = -1
    num_cols = -1
    num_states = -1
    matrix_MDP = None
    reward_per_step = -0.1

    # current x, y position of the agent
    currX = -1
    currY = -1

    # x, y position of a box that can be pushed around
    boxX = -1
    boxY = -1

    # the goal x, y position of the agent
    goalX = -1
    goalY = -1
    # the starting x, y position of the agent
    startX = -1
    startY = -1

    # dimensions for the image to be saved -- initialized in parseMDPString
    # dims_reshape = (10, 10, 3)
    # dimensions for the image to be used by the neural net -- initialized in parseMDPString
    # dims_agent = (3, 10, 10)

    def __init__(self, MDP_path='all-goals/mdps/fig1_randomwalk_box.mdp', cell_size=1):
        if os.path.isfile(MDP_path):
            self.str_MDP = self._readFile(MDP_path)
        else:
            sys.exit('MDP file not found!')

        self._parseMDPString()
        self.num_states = self.num_rows * self.num_cols
        self.cell_size = cell_size

        pygame.init()
        window_title = 'GridWorld Environment'
        window_size = (self.cell_size * self.num_rows, self.cell_size * self.num_cols)
        self.surface = pygame.display.set_mode(window_size, 0, 0)
        # pygame.display.set_caption(window_title)

    def _readFile(self, MDP_path):
        file = open(MDP_path, 'r')
        return_string = ''
        for line in file:
            return_string += line
        return return_string

    def _parseMDPString(self):
        data = self.str_MDP.split('\n')
        self.num_rows = int(data[0].split(',')[0])
        self.num_cols = int(data[0].split(',')[1])
        self.matrix_MDP = np.zeros((self.num_rows, self.num_cols))

        self.dims_agent = (self.num_rows, self.num_cols, 3)
        self.dims_reshape = (self.num_rows, self.num_cols, 3)
        self.action_space = ActionSpace(num_actions=4)
        self.observation_space = ObservationSpace(obs_shape=self.dims_agent)
        # self.action_space = 4
        # self.observation_space = 256

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if data[i + 1][j] == 'X':  # a block in gridworld
                    self.matrix_MDP[i, j] = -1
                elif data[i + 1][j] == '.':  # an eligible state in gridworld
                    self.matrix_MDP[i, j] = 0
                elif data[i + 1][j] == 'B':
                    self.matrix_MDP[i, j] = 1
                    self.boxX = i
                    self.boxY = j
                elif data[i + 1][j] == 'G':  # an eligible state in gridworld
                    self.matrix_MDP[i, j] = 0

    def _makeCanvas(self):
        def makeRect(x, y):
            fill_color = 'white'
            if self.matrix_MDP[x, y] == -1:
                fill_color = 'black'
            if x == self.goalX and y == self.goalY:
                fill_color = 'white'
            if x == self.currX and y == self.currY:
                fill_color = 'red'
            if x == self.boxX and y == self.boxY:
                fill_color = 'blue'
            pygame.draw.rect(self.surface, pygame.Color(fill_color), np.multiply((y, x, y + 1, x + 1), self.cell_size))

        # pygame.draw.rect(self.surface, pygame.Color('black'), np.multiply((y, x, y + 1, x + 1), self.cell_size), 1)

        for x in range(self.num_rows):
            for y in range(self.num_cols):
                makeRect(x, y)

    def _resetEnv(self, random_start=True, i=None, j=None):
        self.currX = i
        self.currY = j

    def generateAllStates(self):
        img_idx = 0
        for k in list(np.arange(6, 8)):
            for m in list(np.arange(2, 8)):
                self.boxX = k
                self.boxY = m
                for i in range(self.num_rows):
                    for j in range(self.num_cols):
                        if self.matrix_MDP[i, j] != -1:
                            if i == self.boxX and j == self.boxY:
                                continue
                            self.currX = i
                            self.currY = j
                            self._makeCanvas()
                            obs = self._getCurrentObservation()
                            img_idx += 1
                            scipy.misc.imsave('gridworld_states/{}.png'.format(img_idx), obs.reshape(self.dims_reshape))

    def _getNextState(self, action):
        # if self.currX == self.goalX and self.currY == self.goalY:
        # 	return self.currX, self.currY, True

        # the state transition function for a box present within the environment

        nextX = self.currX
        nextY = self.currY
        box_nextX = self.boxX
        box_nextY = self.boxY

        if self.matrix_MDP[self.currX, self.currY] != -1:
            if action == 0 and self.currX > 0:
                if self.currX - 1 == self.boxX and self.currY == self.boxY:
                    if self.currX - 2 > 0 and self.matrix_MDP[self.currX - 3, self.currY] != -1:
                        # print('action 0 on box')
                        box_nextX = self.currX - 2
                        box_nextY = self.currY
                else:
                    nextX = self.currX - 1
                    nextY = self.currY
            elif action == 1 and self.currY < self.num_cols - 1:
                if self.currX == self.boxX and self.currY + 1 == self.boxY:
                    if self.currY + 2 < self.num_cols - 1 and self.matrix_MDP[self.currX, self.currY + 3] != -1:
                        # print('action 1 on box')
                        box_nextX = self.currX
                        box_nextY = self.currY + 2
                else:
                    nextX = self.currX
                    nextY = self.currY + 1
            elif action == 2 and self.currX < self.num_rows - 1:
                if self.currX + 1 == self.boxX and self.currY == self.boxY:
                    if self.currX + 2 < self.num_rows - 1 and self.matrix_MDP[self.currX + 3, self.currY] != -1:
                        # print('action 2 on box')
                        box_nextX = self.currX + 2
                        box_nextY = self.currY
                else:
                    nextX = self.currX + 1
                    nextY = self.currY
            elif action == 3 and self.currY > 0:
                if self.currX == self.boxX and self.currY - 1 == self.boxY:
                    if self.currY - 2 > 0 and self.matrix_MDP[self.currX, self.currY - 3] != -1:
                        # print('action 3 on box')
                        box_nextX = self.currX
                        box_nextY = self.currY - 2
                else:
                    nextX = self.currX
                    nextY = self.currY - 1

        if nextX < 0 or nextY < 0:
            sys.exit('There is something wrong in your MDP definition')

        if nextX == self.matrix_MDP.shape[0] or nextY == self.matrix_MDP.shape[1]:
            sys.exit('There is something wrong in your MDP definition')

        # assert not (self.currX == self.goalX and self.currY == self.goalY)
        done = False

        if self.matrix_MDP[nextX, nextY] != -1 and self.matrix_MDP[box_nextX, box_nextY] != -1:
            return nextX, nextY, box_nextX, box_nextY, done
        else:
            return self.currX, self.currY, self.boxX, self.boxY, done

    def _getCurrentObservation(self):
        pygame.display.update()
        img = (imresize(pygame.surfarray.array3d(self.surface).swapaxes(1, 0), self.dims_reshape).reshape(
            self.dims_agent)).astype(np.float32)
        assert img.shape == (self.num_rows, self.num_cols, 3)
        # scipy.misc.imsave('image_generated.png', img.reshape(self.dims_reshape))
        return img

    # def newGame(self, random_start=True):
    #     i = np.random.randint(0, self.num_rows)
    #     j = np.random.randint(0, self.num_cols)
    #     while self.matrix_MDP[i, j] == -1 or self.matrix_MDP[i, j] == 1:
    #         i = np.random.randint(0, self.num_rows)
    #         j = np.random.randint(0, self.num_cols)
    #     self._resetEnv(random_start, i, j)
    #     self._makeCanvas()
    #     return self._getCurrentObservation()

    def reset(self):
        i = np.random.randint(0, self.num_rows)
        j = np.random.randint(0, self.num_cols)
        while self.matrix_MDP[i, j] == -1 or self.matrix_MDP[i, j] == 1:
            i = np.random.randint(0, self.num_rows)
            j = np.random.randint(0, self.num_cols)
        self._resetEnv(random_start=True, i=i, j=j)
        self._makeCanvas()
        return self._getCurrentObservation()

    def act(self, action):
        self.currX, self.currY, self.boxX, self.boxY, done = self._getNextState(action)
        # if done:
        # 	return 0.0, done
        # else:
        # 	return self.reward_per_step, done
        return self.reward_per_step, done

    # def sample(self, action):
    #     reward, done = self.act(action)
    #     self._makeCanvas()
    #     return reward, self._getCurrentObservation(), done

    def step(self, action):
        reward, done = self.act(action)
        self._makeCanvas()
        return self._getCurrentObservation(), reward, done, []

    def close(self):
        del self
        # pass