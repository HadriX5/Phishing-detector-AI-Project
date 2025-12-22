# --------------------------------------------------------------------------------------------------
# phishing_env.py
# --------------------------------------------------------------------------------------------------
# This module defines the PhishingEnv class, which simulates an environment for training
# a Q-learning agent to detect phishing websites based on their features.
# --------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

class PhishingEnv:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.current_index = 0
        self.n_samples = len(X)
        self.n_features = X.shape[1]
    
    def reset(self):
        self.current_index = 0
        return self.X.iloc[self.current_index].values
    
    def step(self, action):
        label = self.y.iloc[self.current_index]
        reward = 1 if action == label else -1
        
        self.current_index += 1
        done = self.current_index >= self.n_samples
        
        if not done:
            next_state = self.X.iloc[self.current_index].values
        else:
            next_state = np.zeros(self.n_features)
            
        return next_state, reward, done