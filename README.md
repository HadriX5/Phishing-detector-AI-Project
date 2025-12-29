# Phishing Detector AI Project

A robust Phishing URL Detector that combines Reinforcement Learning (Q-Learning) for precise pattern memorization and K-Nearest Neighbors (KNN) for generalization of unknown states.

This project addresses the "State Space Explosion" problem in RL by implementing a smart feature discretization pipeline, achieving a 99.56% accuracy on the PhiUSIIL dataset.

## 🧠 The Architecture

Traditional Reinforcement Learning struggles with continuous input data (infinite states). We solved this using a hybrid approach:

-   Discretizer: Converts continuous URL features (e.g., URLCharProb: 0.045) into discrete bins (e.g., Bin: 2).

-   Q-Learning Agent (The Memory): Stores known state-action pairs in a Q-Table. If a URL pattern has been seen before, it returns the optimal classification instantly (O(1)).

-   KNN Fallback (The Generalizer): If the Agent encounters a new, unknown state, it queries a KNN classifier to find the most similar known state and predicts based on neighbors.

## 📊 Key Features

-   🛡️ Hybrid Logic: Combines the speed of Lookup Tables with the flexibility of Supervised Learning.

-   📉 Smart Discretization: Uses Quantile Binning (qcut) to normalize data distributions, preventing outliers from breaking the model.

-   🚫 No Data Leakage: Strict separation of Train/Test sets before discretization ensuring valid real-world performance.

-   📂 Real-World Dataset: Trained on PhiUSIIL, utilizing advanced linguistic and structural features.

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/HadriX5/Phishing-detector-AI-Project.git
cd your-dir
```

2. Create a virtual environment (Optional but recommended):

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## ⚙️ Usage

1. Training the Agent
   To train the Q-Learning agent and the KNN fallback model from scratch:

```bash
python -m scripts.train_rl
    Input: data/features/selected_features_df.parquet
    Output: Saves trained models to data/objects/
```

2. Making Predictions
   To test the model with new, custom URLs (simulated via features):

```bash
python -m scripts.predict_url # hardcoded URLs in code
python -m scripts.functional_predictor # to type in a simple GUI
```

This script loads the saved .pkl models and performs inference on sample legitimate and phishing feature vectors.

## 📈 Performance Results

The model was evaluated on a held-out Test Set (20% of the dataset, approx 40k samples).

| Metric                   | Value      |
| :----------------------- | :--------- |
| **Accuracy**             | **99.56%** |
| **Precision (Phishing)** | 1.00       |
| **Recall (Phishing)**    | 1.00       |
| **F1-Score**             | 1.00       |

### Model Usage during Inference

-   Q-Table Hits (Memory): ~48% (Exact pattern match found)

-   KNN Fallback (Generalization): ~52% (Nearest neighbor search used)

Note: We confirmed via study that removing the KNN fallback drops accuracy to ~73%, proving the necessity of the hybrid approach.

## 🧬 Dataset & Features

The model uses 20 high-impact features extracted from the URL and HTML content. Some notable features include:

-   URLCharProb: Probability of character sequences (trigrams) matching natural language. Random strings (e.g., xkqz.com) score low.

-   CharContinuationRate: Measures the smoothness of character type transitions (Letter-Letter vs Letter-Number-Symbol).

-   NoOfEmptyRef: Count of broken/empty links (`<a href="#">`) on the page. High numbers indicate a "facade" website typical of phishing kits.

-   DomainTitleMatchScore: Similarity score between the page Title and the Domain name.

## 📂 Project Structure

```Plaintext
phishing-detector-rl/
├── data/
│   ├── features/          # Raw parquet datasets
│   └── objects/           # Saved models (.pkl)
├── notebooks/             # Jupyter notebooks for EDA
├── scripts/
│   ├── train_rl.py        # Main training loop
│   └── predict_url.py     # Inference script
├── src/
│   ├── models/
│   │   └── q_agent.py     # QLearningAgent Class with KNN logic
│   ├── utils/
│   │   └── discretizer.py # Discretization logic (Fit/Transform)
│   └── environment/
│       └── phishing_env.py # RL Environment
├── requirements.txt
└── README.md
```

## 📝 Authors

-   Adrià Suárez $\to$ [GitHub](https://github.com/HadriX5)

-   Albert Marin $\to$ [GitHub](https://github.com/MalbertMB)

-   Izan Farreny $\to$ [GitHub](https://github.com/IzanFarreny)

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
