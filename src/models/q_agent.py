# --------------------------------------------------------------------------------------------------
# q_agent.py
# --------------------------------------------------------------------------------------------------
# This module defines a Q-learning agent that uses a K-Nearest Neighbors
# classifier when it encounters an unknown state.
# --------------------------------------------------------------------------------------------------
# Date: December 19, 2025
# Author: A very tired student programming at 2 AM
# --------------------------------------------------------------------------------------------------

import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

class QLearningAgent:
    """
    A Q-learning agent with KNN fallback for unknown states.
    1. Uses a dictionary-based Q-table to store state-action values.
    2. Employs an epsilon-greedy strategy for action selection.
    3. Falls back to a KNN classifier when encountering unknown states during testing.

    Parameters:
    - action_space_size: Number of possible actions.
    - learning_rate: Rate at which the agent learns.
    - discount_factor: Discount factor for future rewards.
    - epsilon: Initial exploration rate.
    - epsilon_decay: Decay rate for epsilon after each episode.
    - min_epsilon: Minimum value for epsilon.
    - knn_neighbors: Number of neighbors for the KNN classifier.

    Methods:
    - choose_action(state, is_test): Selects an action based on the current state.
    - learn(state, action, reward, next_state): Updates the Q-table based on experience
    - decay_epsilon(): Decays the exploration rate.
    - train_knn(X_train, y_train): Trains the KNN classifier.
    - save_agent(filepath): Saves the agent's state to a file.
    - load_agent(filepath): Loads the agent's state from a file.
    """

    def __init__(self, 
                action_space_size = 2, 
                learning_rate = 0.1, 
                discount_factor = 0.9, 
                epsilon = 1.0, 
                epsilon_decay = 0.99, 
                min_epsilon = 0.1, 
                knn_neighbors = 5):
        
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}
        self.actions = list(range(action_space_size))

        # KNN Classifier fallback
        self.knn_fallback = KNeighborsClassifier(n_neighbors = knn_neighbors)

        # Purely for debug purposes
        self.q_hits = 0
        self.knn_hits = 0

    def get_state_key(self, state):
        """
        Converts a state array into a hashable key for the Q-table.

        Args:
            state (np.ndarray): The current state.
        Returns:
            tuple: A hashable representation of the state.
        """
        if state is None:
            return None
        
        return tuple(state.astype(int))
    
    def train_knn(self, X_train, y_train):
        """
        Trains the KNN classifier on the provided training data.
        Args:
            X_train (np.ndarray or pd.DataFrame): Training features.
            y_train (np.ndarray or pd.Series): Training labels.
        Returns:
            None
        """

        self.knn_fallback.fit(X_train, y_train)
    
    def choose_action(self, state, is_test = False):
        """
        Chooses an action based on the current state using an epsilon-greedy strategy.
        Args:
            state (np.ndarray): The current state.
            is_test (bool): Flag indicating if the agent is in test mode.
        Returns:
            int: The chosen action.
        """

        state_key = self.get_state_key(state)

        # If state not in Q-table, this is False, so we can use KNN
        is_known_state = any((state_key, a) in self.q_table for a in self.actions)

        if is_test:
            if not is_known_state:
                # Purely for debug purposes
                self.knn_hits += 1

                return self.knn_fallback.predict([state])[0]
            
            # Purely for debug purposes
            self.q_hits += 1

            q_values = [self.q_table.get((state_key, a), 0.0) for a in self.actions]
            return int(np.argmax(q_values))
        
        # -- Exploration vs Exploitation --
        if np.random.rand() < self.epsilon or not is_known_state:
            return np.random.choice(self.actions)
        
        q_values = [self.q_table.get((state_key, a), 0.0) for a in self.actions]
        return int(np.argmax(q_values))
    
    def decay_epsilon(self):
        """
        Decays the exploration rate epsilon towards min_epsilon.
        """

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def learn(self, state, action, reward, next_state):
        """
        Updates the Q-table based on the agent's experience using the Bellman equation.
        Args:
            state (np.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state after taking the action.
        Returns:
            None
        """

        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        current_q = self.q_table.get((state_key, action), 0.0)
        max_future_q = max([self.q_table.get((next_state_key, a), 0.0) for a in self.actions])

        # Bellman Equation
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_future_q - current_q)
        self.q_table[(state_key, action)] = new_q

    def save_agent(self, filepath):
        """
        Saves the entire agent instance (including Q-Table and KNN).
        
        Args:
            filepath (str): Path to the file where the agent will be saved.
        Returns:
            None
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """
        Loads the entire agent instance (including Q-Table and KNN).
        
        Args:
            filepath (str): Path to the file from which the agent will be loaded.
        Returns:
            QLearningAgent: The loaded agent instance.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)