# --------------------------------------------------------------------------------------------------
# train_rl.py
# --------------------------------------------------------------------------------------------------
# This module implements the training loop for the Q-learning agent 
# with KNN fallback described in q_agent.py.
# --------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from q_agent import QLearningAgent # TO BE CHANGED WHEN MERGED WITH MAIN & DATA_PROCESSING branches
from phishing_env import PhishingEnv

# -- CONFIGURATION PARAMETERS ----------------------------------------------------------------------
DATA_PATH = "large_dummy_dataset.parquet" # TO BE CHANGED 2
MODEL_PATH = "q_agent_knn.pkl"  # TO BE CHANGED 2

EPISODES = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.90
MIN_EPSILON = 0.01
KNN_NEIGHBORS = 5
# --------------------------------------------------------------------------------------------------

def main():
    # Load data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"No s'ha trobat el fitxer: {DATA_PATH}")
    
    df = pd.read_parquet(DATA_PATH)

    y = df['label']
    X = df.drop(columns = ['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = TEST_SIZE, 
                                                        random_state = RANDOM_STATE)
    
    env = PhishingEnv(X_train, y_train)
    agent = QLearningAgent(
        action_space_size = 2,
        learning_rate = ALPHA,
        discount_factor = GAMMA,
        epsilon = EPSILON,
        epsilon_decay = EPSILON_DECAY,
        min_epsilon = MIN_EPSILON,
        knn_neighbors = KNN_NEIGHBORS)
    
    for episode in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state, is_test = False)
            next_state, reward, done = env.step(action)

            agent.learn(state, action, reward, next_state)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
    
    # Train KNN on all known state-action pairs
    agent.train_knn(X_train.values, y_train)

    predictions = []

    # Reset debug counters
    agent.q_hits = 0
    agent.knn_hits = 0

    test_states = X_test.values
    for state in test_states:
        action = agent.choose_action(state, is_test = True)
        predictions.append(action)

    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print("-" * 100)
    print(f"RESULTATS FINALS")
    print("-" * 100)
    print(f"ACCURACY: {acc:.4f}")
    print(f"\nMATRIU DE CONFUSIÓ:\n{cm}")
    print(f"\nINFORME DETALLAT:\n{classification_report(y_test, predictions)}")
    
    print("-" * 100)
    print("ÚS DELS MODELS DURANT EL TEST:")
    total_preds = agent.q_hits + agent.knn_hits
    if total_preds > 0:
        print(f"Q-Table Hits: {agent.q_hits} ({agent.q_hits/total_preds:.1%})")
        print(f"KNN Fallback: {agent.knn_hits} ({agent.knn_hits/total_preds:.1%})")

    # Save the trained agent
    agent.save_agent(MODEL_PATH)

if __name__ == "__main__":
    main()