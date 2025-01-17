import logging
import os

import pandas as pd
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

from sklearn.metrics import confusion_matrix, cohen_kappa_score
from itertools import combinations
import numpy as np


def disagreement_measure(predictions):
    disagreements = 0
    total_pairs = 0

    for pred1, pred2 in combinations(predictions, 2):
        disagreements += np.sum(pred1 != pred2)
        total_pairs += len(pred1)

    return disagreements / total_pairs


def q_statistic(pred1, pred2):
    cm = confusion_matrix(pred1, pred2)
    q = (cm[0, 0] * cm[1, 1] - cm[0, 1] * cm[1, 0]) / (
        cm[0, 0] * cm[1, 1] + cm[0, 1] * cm[1, 0] + 1e-10
    )
    return q


def analyze_diversity(ensemble, X, transform=True):
    predictions = []
    X_np = X.to_numpy() if hasattr(X, 'to_numpy') and transform else X

    if hasattr(ensemble, 'estimators_'):
        for model in ensemble.estimators_:
            predictions.append(model.predict(X_np))
    else:
        logging.error("Ensemble type not supported for diversity analysis.")
        return {}

    disagreement = disagreement_measure(predictions)

    q_stats = []
    kappas = []
    for pred1, pred2 in combinations(predictions, 2):
        q_stats.append(q_statistic(pred1, pred2))
        kappas.append(cohen_kappa_score(pred1, pred2))

    avg_q_stat = np.mean(q_stats)
    avg_kappa = np.mean(kappas)

    return {
        'Disagreement': disagreement,
        'Average Q-Statistic': avg_q_stat,
        'Average Kappa': avg_kappa
    }

def train_ensembles(ensembles, X_train, y_train):
    for i, ensemble in enumerate(ensembles):
        logging.info(f"Training {i + 1}/{len(ensembles)} ensembles...")
        ensemble.fit(X_train, y_train)
    return ensembles


if __name__ == '__main__':
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv('./data/telecom_churn.csv')

    X = data.drop(columns=['Churn'])
    y = data['Churn']

    rf = RandomForestClassifier(random_state=42, n_estimators=5)
    gb = GradientBoostingClassifier(random_state=42)
    dt = DecisionTreeClassifier(random_state=42)
    logreg = LogisticRegression(random_state=42, max_iter=1000)

    ensemble_homogeneous = RandomForestClassifier(n_estimators=20, random_state=42)
    ensemble_heterogeneous = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('logreg', logreg)], voting='soft'
    )
    adaboost = AdaBoostClassifier(n_estimators=20, random_state=42)
    easy_ensemble = EasyEnsembleClassifier(n_estimators=20, random_state=42)
    ensembles = [
        ensemble_homogeneous, ensemble_heterogeneous, adaboost, easy_ensemble
    ]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_ensembles(ensembles, X_train, y_train)

    logging.info("Testing Homogeneous Ensemble...")
    homogeneous_diversity = analyze_diversity(ensemble_homogeneous, X)

    logging.info("Testing Heterogeneous Ensemble...")
    heterogeneous_diversity = analyze_diversity(ensemble_heterogeneous, X, transform=False)

    logging.info("Testing AdaBoost...")
    ada_diversity = analyze_diversity(adaboost, X, transform=True)

    logging.info("Testing Easy Ensemble...")
    easy_diversity = analyze_diversity(easy_ensemble, X, transform=True)

    metrics = "Homogeneous Ensemble Diversity Metrics: "
    for metric, value in homogeneous_diversity.items():
        metrics += f"{metric}: {value:.4f}; "
    logging.info(metrics)

    metrics = "Heterogeneous Ensemble Diversity Metrics: "
    for metric, value in homogeneous_diversity.items():
        metrics += f"{metric}: {value:.4f}; "
    logging.info(metrics)

    metrics = "AdaBoost Diversity Metrics: "
    for metric, value in ada_diversity.items():
        metrics += f"{metric}: {value:.4f}; "
    logging.info(metrics)

    metrics = "Easy Ensemble Diversity Metrics: "
    for metric, value in easy_diversity.items():
        metrics += f"{metric}: {value:.4f}; "
    logging.info(metrics)

    csv_data = [
        {'Ensemble': 'Homogeneous', **homogeneous_diversity},
        {'Ensemble': 'Heterogeneous', **heterogeneous_diversity},
        {'Ensemble': 'AdaBoost', **ada_diversity},
        {'Ensemble': 'Easy Ensemble', **easy_diversity}
    ]

    results_df = pd.DataFrame(csv_data)
    results_df.to_csv(os.path.join(output_dir, "results_diverity.csv"), index=False)

    logging.info("Diversity metrics saved to 'results_diversity.csv'")