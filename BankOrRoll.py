# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:37:13 2023

@author: sahil
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces


class BankOrRoll(gym.Env):
    """
        The Idea of the game is simple:
            You go first.
            You can either Bank or Roll
                Bank will increase your points by your bank level
                Roll will doll a {dice_sides}-dice
                    If the Roll is a {death_roll}
                        You loose everything in the bank
                    Else
                        Bank goes up by the Roll
            Gaol is to get Points >= {terminal_level}
    
    """
    def __init__(self, terminal_level=100,  
                 dice_sides=6, death_roll=6,
                 opponent_policy=None,):

        self.terminal_level = terminal_level
        self.death_roll = death_roll-1
        self.dice_roll_space = spaces.Discrete(dice_sides)

        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), 
                                    high=np.array([terminal_level, 
                                                   terminal_level, 
                                                   terminal_level+1]), 
                                    dtype=np.int8)

        # 0 - Bank; 1 - Roll;
        self.action_space = spaces.Discrete(2)

        self.plyr_points = 0
        self.oppo_points = 0
        self.plyr_bank = 0
        self.oppo_bank = 0
        
        if opponent_policy is None:
            self.opponent_policy = lambda oppo_points, plyr_points, oppo_bank : self.action_space.sample()
        else:
            self.opponent_policy = opponent_policy
    
    def _get_obs(self):
        return np.array([self.plyr_points, self.oppo_points, self.plyr_bank])
    
    def _get_info(self):
        return {'Player Points': self.plyr_points,
                'Opponent Points': self.oppo_points,
                'Player Bank': self.plyr_bank,
                'Opponent Bank': self.oppo_bank}
    
    def reset(self, ):
        self.plyr_points = 0
        self.oppo_points = 0
        self.plyr_bank = 0
        self.oppo_bank = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def _simulate_opponent(self):
        opp_action = self.opponent_policy(
                oppo_points=self.oppo_points, 
                plyr_points=self.plyr_points, 
                oppo_bank=self.oppo_bank )
        # Bank
        if opp_action == 0:
            self.oppo_points += self.oppo_bank
            self.oppo_bank = 0
            if self.oppo_points >= self.terminal_level:
                return True
            else:
                return False
        # Roll
        elif opp_action == 1:
            dice_roll = self.dice_roll_space.sample()
            if dice_roll == self.death_roll:
                self.oppo_bank = 0
                return False
            else:
                self.oppo_bank += dice_roll
                self.oppo_bank = min(self.oppo_bank, self.terminal_level)
                return self._simulate_opponent()
                
                

    def step(self, action):
        reward = 0
        terminated = False
        # Bank
        if action == 0:
            reward =self.plyr_bank
            self.plyr_points += self.plyr_bank
            self.plyr_bank = 0
            if self.plyr_points >= self.terminal_level:
                terminated = True
                reward = 100
            else:
                # Simulate Opponent's move
                terminated = self._simulate_opponent()
                
                if terminated:
                    reward = -100
                    
                
        # Roll
        elif action == 1:
            dice_roll = self.dice_roll_space.sample()
            if dice_roll == self.death_roll:
                reward = -1*self.plyr_bank
                self.plyr_bank = 0
            else:
                reward = dice_roll
                self.plyr_bank += dice_roll
                self.plyr_bank = min(self.plyr_bank, self.terminal_level)

            
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info
    


if __name__ == '__main__':
    NUM_EPISODES = 100000
    LEARN_RATE = 0.0003
    DISCOUNT = 0.9
    
    env = BankOrRoll(terminal_level=30)
    
    policy = np.ones(env.observation_space.high)*0.5

    for episode_idx in range(NUM_EPISODES):
        observation, info = env.reset()
        state_vector = []
        action_vector = []
        terminated = False
        while not terminated:
            state_vector.append(observation)
            #action = env.action_space.sample()
            action = int(bool(np.random.random() < policy[tuple(observation)]))
            action_vector.append(action)

            observation, reward, terminated, truncated, info = env.step(action)
        
        if episode_idx % 1000 == 0:
            print(episode_idx, reward)
            sns.heatmap(policy[:, :, 0], square=True, vmin=0, vmax=1, cmap='coolwarm')
            plt.show()

        for action, state in zip(action_vector[::-1], state_vector[::-1]):
            policy[tuple(state)] += action*reward*LEARN_RATE - (1-action)*reward*LEARN_RATE
            policy[tuple(state)] = min(max(policy[tuple(state)], 0),1)
            reward = reward*DISCOUNT
    
        # Update the Policy for opp
        env.opponent_policy = lambda oppo_points, plyr_points, oppo_bank: int(bool(np.random.random() < policy[oppo_points, plyr_points, oppo_bank]))
    
    
        