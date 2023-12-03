# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 18:13:37 2023

@author: sahil
"""
import numpy as np

state_space = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
])

action_space = {
       0: np.array([-1, 0]),    # UP
       1: np.array([0, 1]),     # RIGHT
       2: np.array([1, 0]),     # DOWN
       3: np.array([0, -1]),    # LEFT
   }




COUNT_VECTOR = np.zeros((4,4))
REWARD_VECTOR = np.zeros((4,4))

def run_episode(starting_idx):
    
    state_vector = [starting_idx]
    update_vector = [True]
    reward_vector = []
    _idx = starting_idx
    done = bool(state_space[ _idx[0], _idx[1] ] == 1)
    while not done:
        
        action = action_space[np.random.randint(low=0, high=4)]
        _idx = np.clip(_idx+action, 0, 3)
        done = bool(state_space[ _idx[0], _idx[1] ] == 1)
        update_vector.append( ~np.any([np.all(_idx == s) for s in state_vector]) 
            )
        state_vector.append(_idx)
        reward_vector.append(-1)
    reward_vector.append(0)
    
    s_list = np.array(state_vector)[update_vector]
    r_list = np.cumsum(reward_vector[::-1])[::-1][update_vector]
    
    for s, r in zip(s_list, r_list):
        COUNT_VECTOR[s[0], s[1]] += 1
        REWARD_VECTOR[s[0], s[1]] += r

for i in range(1_000_000):
    starting_idx = np.array([np.random.randint(low=0, high=4),
                             np.random.randint(low=0, high=4)])
    run_episode(starting_idx)
    if i % 100_000 == 0:
        print(np.round(REWARD_VECTOR/COUNT_VECTOR, 2))
    
