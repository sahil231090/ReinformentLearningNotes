# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 00:23:24 2023

@author: sahil
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces
from copy import copy




class GridWorld(gym.Env):
    """
        The Idea of the game is simple:
            You are given a {length}X{height} grid
            Each value in the array is either
                0 Free Space
                1 Player Location
                2 Goal location
                3 Terminal State
            You start with a {initial_budget}
            You can either go
                0 Right, 
                1 Left,
                2 Down, 
                3 Up, 
            Each move costs 1 {move_cost}
            The game can run for a max of {max_player_turns}
            Hitting a the Terminal state will end the game, you get {negative_reward}
            Reaching the goal give you {initial_budget} - total {move_cost}/s + {positive_reward}
    
    """
    def __init__(self, length=4, height=4,
                 initial_budget=100, move_cost=1,
                 max_player_turns=100, 
                 negative_reward=-10, positive_reward=+10
                 ):
        
        self.FREE_SPACE = 0
        self.PLAYER_LOCATION = 1
        self.GOAL_LOACTION = 2
        self.TERMINAL_STATE = 3
        
        self.length = length
        self.height = height
        
        self.max_player_turns = max_player_turns
        self.negative_reward = negative_reward
        self.positive_reward = positive_reward
        

        self.observation_space = spaces.MultiDiscrete([length, height, 4])
        self.action_space = spaces.Discrete(4)
        self.state = None
        
        self.reset()
    
    
    def _has_valid_path(self, state, tidx, tjdx, idx, jdx):
        state_candidate = copy(state)
        state_candidate[tidx, tjdx] = self.TERMINAL_STATE
        
        def _dfs(cidx, cjdx, state_candidate):
            # Check if the current position is out of bounds or a wall
            if (cidx < 0) or \
               (cjdx < 0) or \
               (cidx >= self.length) or \
               (cjdx >= self.height) or \
               state_candidate[cidx, cjdx]  == self.TERMINAL_STATE:
                return False
            
            # Check if the current position is the target
            if state_candidate[cidx][cjdx] == self.GOAL_LOACTION:
                return True
            
            # Mark the current position as visited
            state_candidate[cidx][cjdx] = self.TERMINAL_STATE  # Marked as visited
            
            # Explore all four possible directions: up, down, left, right
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dr, dc in directions:
                if _dfs(cidx + dr, cjdx + dc, state_candidate):
                    return True

            return False
        del state_candidate
        
    
    def reset(self):
        # Create all zeros
        self.state = np.zeros((self.length, self.height), dtype=np.int8)
        
        # Pick a random starting point
        idx, jdx = np.random.choice(self.length), np.random.choice(self.height)
        self.state[idx, jdx]= self.PLAYER_LOCATION

        
        # Pick a Goal
        gidx, gjdx =  np.random.choice(self.length), np.random.choice(self.height)
        # Make sure it is not the same as the player index
        while (gidx, gjdx) == (idx, jdx):
            gidx, gjdx =  np.random.choice(self.length), np.random.choice(self.height)
        self.state[gidx, gjdx]= self.GOAL_LOACTION
        

        num_terminal_states = np.int(np.sqrt(self.length*self.height))
        for i in range(num_terminal_states):
            added_flag = False
            while not added_flag:
                tidx, tjdx =  np.random.choice(self.length), np.random.choice(self.height)
                while ((tidx, tjdx) == (idx, jdx)) or ((tidx, tjdx) == (gidx, gjdx)):
                    tidx, tjdx =  np.random.choice(self.length), np.random.choice(self.height)
                if i < min(self.length, self.height) or self._has_valid_path(self.state, tidx, tjdx, idx, jdx):
                    self.state[tidx, tjdx] = self.TERMINAL_STATE
                    
                    added_flag = True

        self.turns_left = self.max_player_turns
        self.turns_made = 0
        self.move_history = []
        self.current_location = (idx, jdx)
        

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info            
        
    
    def _get_obs(self):
        return self.state
    
    def _get_info(self):
        return {'Player Points': self.turns_left,
                'Opponent Points': self.turns_made,
                'Move History': self.move_history,
                'Current Locaiton': self.current_location}

    def step(self, action):
        reward = 0
        terminated = False
        
        assert action in [0,1,2,3]
        
        dx, dy = [(0, 1), (0, -1), (1, 0), (-1, 0)][action]
        
        x = np.clip(self.current_location[0]+dx, 0, self.length-1)
        y = np.clip(self.current_location[1]+dy, 0, self.height-1)

        new_location = (x,y)
        
        self.turns_made += 1
        self.turns_left -= 1
        
        if self.turns_left == 0:
            reward = self.negative_reward
            terminated = True
        elif self.state[new_location[0], new_location[1]] == self.TERMINAL_STATE:
            reward = self.negative_reward
            terminated = True
        elif self.state[new_location[0], new_location[1]] == self.GOAL_LOACTION:
            reward = self.positive_reward + self.turns_left
            terminated = True
        else:
            reward = 0
            terminated = False
            self.state[new_location[0], new_location[1]] = 1
            self.state[self.current_location[0], self.current_location[1]] = 0
            self.current_location = new_location
            
        self.move_history.append(action)
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info
    
if __name__ == '__main__':
    NUM_EPISODES = 10
    
    env = GridWorld()
    for episode_idx in range(NUM_EPISODES):
        observation, info = env.reset()
        
        terminated = False
        while not terminated:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
        print(episode_idx, reward)

    
