# --------------------------------------------------------------------------------------------------
# train_rl.py
# --------------------------------------------------------------------------------------------------
# This module implements the training loop for the Q-learning agent 
# with KNN fallback described in q_agent.py.
# --------------------------------------------------------------------------------------------------
import sys

import pandas as pd
import numpy as np
import os
import pickle

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src import QLearningAgent, PhishingEnv, Discretizer


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# -- CONFIGURATION PARAMETERS ----------------------------------------------------------------------
DATA_PATH = "data/features/selected_features_df.parquet" 
MODEL_PATH = "data/objects/q_agent_knn.pkl"
DISC_PATH = "data/objects/discretizer.pkl"

EPISODES = 600 # 1000 episodes = 16h of training on a standard PC, do tests with 5,10 or 50 first
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_BINS = 5

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.1
KNN_NEIGHBORS = 5
# --------------------------------------------------------------------------------------------------

def main():
    # Load raw data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"No s'ha trobat el fitxer: {DATA_PATH}")
    
    df = pd.read_parquet(DATA_PATH)

    cols_to_drop = ['URL', 'Domain', 'TLD', 'Title']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns] + ['label'])
    y = df['label']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size = TEST_SIZE, random_state = RANDOM_STATE, stratify = y
    )

    discretizer = Discretizer(n_bins = N_BINS)
    
    discretizer.fit(X_train_raw)
    
    X_train = discretizer.transform(X_train_raw)
    X_test = discretizer.transform(X_test_raw)

    # Save the discretizer
    with open(DISC_PATH, 'wb') as f:
        pickle.dump(discretizer, f)

    
    env = PhishingEnv(X_train, y_train)
    agent = QLearningAgent(
        action_space_size = 2,
        learning_rate = ALPHA,
        discount_factor = GAMMA,
        epsilon = EPSILON,
        epsilon_decay = EPSILON_DECAY,
        min_epsilon = MIN_EPSILON,
        knn_neighbors = KNN_NEIGHBORS)

    # For debug purposes
    episode_rewards = []
    episode_deltas = []

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0

        total_episode_delta = 0
        steps_in_episode = 0

        while not done:
            action = agent.choose_action(state, is_test = False)
            next_state, reward, done = env.step(action)

            delta = agent.learn(state, action, reward, next_state)
            total_episode_delta += delta
            steps_in_episode += 1

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        avg_delta = total_episode_delta / max(1, steps_in_episode)
        episode_deltas.append(avg_delta)
        
        print(f"Ep {episode + 1}/{EPISODES} | Reward: {total_reward} | Avg Delta: {avg_delta:.4f} | Epsilon: {agent.epsilon:.3f}")

        agent.decay_epsilon()

    plt.figure(figsize=(10, 6))
    
    # Dades reals
    plt.plot(range(1, EPISODES + 1), episode_deltas, color = 'red', alpha=0.4, label='Delta Q (Raw)')
    
    # Suavitzat (per veure millor la tendència)
    if len(episode_deltas) > 20:
        deltas_smooth = pd.Series(episode_deltas).rolling(window=20).mean()
        plt.plot(range(1, EPISODES + 1), deltas_smooth, color='darkred', linewidth=2, 
                 label = 'Delta Q (Smoothed)')

    plt.xlabel('Episodi')
    plt.ylabel('Average Absolute TD-Error')
    plt.title('Convergència Matemàtica (Q-Value Stability)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.savefig('data/others/convergence_delta.png')
    plt.close()
    
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