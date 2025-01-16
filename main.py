import logging
from multiprocessing import Pool

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTEN, SVMSMOTE, KMeansSMOTE, BorderlineSMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from collections import Counter


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

from sklearn.metrics import confusion_matrix, cohen_kappa_score
from itertools import combinations
import numpy as np


def cluster_based_oversampling(X, y, n_clusters=10, random_state=42):
    minority_class = y.value_counts().idxmin()
    majority_class = y.value_counts().idxmax()

    X_minority = X[y == minority_class]
    y_minority = y[y == minority_class]

    X_majority = X[y == majority_class]
    y_majority = y[y == majority_class]

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(X_minority)

    oversampled_X = []
    oversampled_y = []

    for cluster in np.unique(clusters):
        cluster_samples = X_minority[clusters == cluster]
        n_samples_to_generate = len(X_majority) // n_clusters - len(cluster_samples)
        if n_samples_to_generate > 0:
            sampled_indices = np.random.choice(cluster_samples.index, n_samples_to_generate, replace=True)
            oversampled_X.append(cluster_samples.loc[sampled_indices])
            oversampled_y.append([minority_class] * n_samples_to_generate)

    oversampled_X = pd.concat([X_majority] + oversampled_X)
    oversampled_y = pd.concat([y_majority] + [pd.Series(ys) for ys in oversampled_y])

    return oversampled_X, oversampled_y


def weighted_random_oversample(X, y):
    minority_class = y.value_counts().idxmin()
    majority_class = y.value_counts().idxmax()

    majority_X = X[y == majority_class]
    majority_y = y[y == majority_class]

    minority_X = X[y == minority_class]
    minority_y = y[y == minority_class]

    minority_X_oversampled = minority_X.sample(len(majority_y), replace=True, random_state=42)
    minority_y_oversampled = minority_y.sample(len(majority_y), replace=True, random_state=42)

    X_balanced = pd.concat([majority_X, minority_X_oversampled])
    y_balanced = pd.concat([majority_y, minority_y_oversampled])

    return X_balanced, y_balanced


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
        cm[0, 0] * cm[1, 1] + cm[0, 1] * cm[1, 0]
    )
    return q


def analyze_diversity(ensemble, X, y):
    logging.info("Analyzing diversity...")

    predictions = []
    for name, model in ensemble.estimators_:
        predictions.append(model.predict(X))

    disagreement = disagreement_measure(predictions)

    q_stats = []
    kappas = []
    for pred1, pred2 in combinations(predictions, 2):
        q_stats.append(q_statistic(pred1, pred2))
        kappas.append(cohen_kappa_score(pred1, pred2))

    avg_q_stat = np.mean(q_stats)
    avg_kappa = np.mean(kappas)

    logging.info(f"Disagreement Measure: {disagreement:.4f}")
    logging.info(f"Average Q-Statistic: {avg_q_stat:.4f}")
    logging.info(f"Average Cohen's Kappa: {avg_kappa:.4f}")

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



def test_balancing(modifier, models, X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    results = {}

    for name, model in models.items():
        logging.info(f"Testing with model: {name}")
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        aucs = []

        for train_index, test_index in cv.split(X, y):
            logging.info(f"Fold {len(accuracies) + 1}/{cv.get_n_splits(X, y)}")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # X_train_smote, y_train_smote = modifier.fit_resample(X_train, y_train)

            if callable(modifier):
                if modifier.__name__ == "generate_gan_samples":
                    X_generated, y_generated = modifier(X_train, y_train, len(y_train))
                    X_train_smote = pd.concat([X_train, X_generated])
                    y_train_smote = pd.concat([y_train, pd.Series(y_generated)])
                else:
                    X_train_smote, y_train_smote = modifier(X_train, y_train)
            else:
                X_train_smote, y_train_smote = modifier.fit_resample(X_train, y_train)

            model.fit(X_train_smote, y_train_smote)
            y_pred = model.predict(X_test)

            accuracies.append(accuracy_score(y_test, y_pred))
            report = classification_report(y_test, y_pred, output_dict=True)
            precisions.append(report['1']['precision'])
            recalls.append(report['1']['recall'])
            f1_scores.append(report['1']['f1-score'])

            if hasattr(model, 'predict_proba'):
                aucs.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

        logging.info(
            f"Model: {name}: Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f}); "
            f"Precision: {np.mean(precisions):.4f} (+/- {np.std(precisions):.4f}); "
            f"Recall: {np.mean(recalls):.4f} (+/- {np.std(recalls):.4f}); "
            f"F1-score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})"
        )

        if aucs:
            logging.info(f"ROC AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")

        results[name] = {
            'Accuracy': np.mean(accuracies),
            'Precision': np.mean(precisions),
            'Recall': np.mean(recalls),
            'F1-score': np.mean(f1_scores),
            'ROC AUC': np.mean(aucs) if aucs else None
        }
    return results


if __name__ == '__main__':
    data = pd.read_csv('./data/creditcard.csv')

    X = data.drop(columns=['Class'])
    y = data['Class']

    rf = RandomForestClassifier(random_state=42, n_estimators=2)
    gb = GradientBoostingClassifier(random_state=42)
    dt = DecisionTreeClassifier(random_state=42)
    logreg = LogisticRegression(random_state=42, solver='saga', max_iter=1000)

    ensemble_homogeneous = RandomForestClassifier(n_estimators=20, random_state=42)
    ensemble_heterogeneous = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('logreg', logreg)], voting='soft'
    )

    models = {
        'Random Forest': rf,
        'Gradient Boosting': gb,
        'Decision Tree': dt,
        'Logistic Regression': logreg,
        'Homogeneous Ensemble': ensemble_homogeneous,
        'Heterogeneous Ensemble': ensemble_heterogeneous
    }

    modifiers = {
        "SMOTE": SMOTE(random_state=42),
        "SMOTEN": SMOTEN(random_state=42),
        "SVMSMOTE": SVMSMOTE(random_state=42),
        "KMeansSMOTE": KMeansSMOTE(random_state=42),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=42)
    }

    end_results = {}

    for mod_name, modifier in modifiers.items():
        logging.info(f"Testing {mod_name}")
        end_results[mod_name] = test_balancing(modifier, models, X, y)

    logging.info("\n" + 20 * "=" + "Summary:")
    for name, part_dict in end_results.items():
        logging.info(f"\n{name}")
        for name, metrics in part_dict.items():
            logging.info(f"Model: {name}")
            logging.info(
                f"Accuracy: {metrics['Accuracy']:.4f}; "
                f"Precision: {metrics['Precision']:.4f}; "
                f"Recall: {metrics['Recall']:.4f}; "
                f"F1-score: {metrics['F1-score']:.4f}"
            )
            if metrics['ROC AUC'] is not None:
                logging.info(f"ROC AUC: {metrics['ROC AUC']:.4f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    ensembles = train_ensembles([ensemble_homogeneous, ensemble_heterogeneous], X_train, y_train)

    logging.info("Diversity Analysis for Homogeneous Ensemble")
    homogeneous_diversity = analyze_diversity(ensemble_homogeneous, X, y)

    logging.info("Diversity Analysis for Heterogeneous Ensemble")
    heterogeneous_diversity = analyze_diversity(ensemble_heterogeneous, X, y)

    logging.info("\n" + 20 * "=" + " Diversity Summary:")
    logging.info("Homogeneous Ensemble Diversity Metrics:")
    for metric, value in homogeneous_diversity.items():
        logging.info(f"{metric}: {value:.4f}")

    logging.info("Heterogeneous Ensemble Diversity Metrics:")
    for metric, value in heterogeneous_diversity.items():
        logging.info(f"{metric}: {value:.4f}")



