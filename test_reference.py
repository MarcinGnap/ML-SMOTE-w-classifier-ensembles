import logging
import os

import numpy as np
import pandas as pd
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def evaluate_model_with_logging(result_csv_path, name, model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    aucs = []

    for train_index, test_index in cv.split(X, y):
        logging.info(f"Fold {len(accuracies) + 1}/{cv.get_n_splits(X, y)}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

        if hasattr(model, 'predict_proba'):
            aucs.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    mean_metrics = {
        'Accuracy': np.mean(accuracies),
        'Accuracy (std)': np.std(accuracies),
        'Precision': np.mean(precisions),
        'Precision (std)': np.std(precisions),
        'Recall': np.mean(recalls),
        'Recall (std)': np.std(recalls),
        'F1-score': np.mean(f1_scores),
        'F1-score (std)': np.std(f1_scores),
        'ROC AUC': np.mean(aucs) if aucs else None,
        'ROC AUC (std)': np.std(aucs) if aucs else None,
    }

    logging.info(
        f"Model: {name}: Accuracy: {mean_metrics['Accuracy']:.4f} (+/- {np.std(accuracies):.4f}); "
        f"Precision: {mean_metrics['Precision']:.4f} (+/- {np.std(precisions):.4f}); "
        f"Recall: {mean_metrics['Recall']:.4f} (+/- {np.std(recalls):.4f}); "
        f"F1-score: {mean_metrics['F1-score']:.4f} (+/- {np.std(f1_scores):.4f})"
        f"ROC AUC: {mean_metrics['ROC AUC']:.4f} (+/- {np.std(aucs):.4f})" if aucs else ""
    )

    result_row = {
        # "Modifier": modifier.__class__.__name__,
        "Model": name,
        'Accuracy': np.mean(accuracies),
        'Accuracy (std)': np.std(accuracies),
        'Precision': np.mean(precisions),
        'Precision (std)': np.std(precisions),
        'Recall': np.mean(recalls),
        'Recall (std)': np.std(recalls),
        'F1-score': np.mean(f1_scores),
        'F1-score (std)': np.std(f1_scores),
        'ROC AUC': np.mean(aucs) if aucs else None,
        'ROC AUC (std)': np.std(aucs) if aucs else None
    }
    pd.DataFrame([result_row]).to_csv(result_csv_path, mode='a', header=not pd.io.common.file_exists(result_csv_path),
                                      index=False)

    return mean_metrics


if __name__ == '__main__':
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    directory = 'data'

    label1 = ['1_telecom_churn.csv']

    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    for file in csv_files:
        logging.info(f"Starting evaluation for {file}")
        file_path = os.path.join(directory, file)

        data = pd.read_csv(file_path)

        if file in label1:
            X = data.drop(columns=data.columns[0])
            y = data.iloc[:, 0]
        else:
            X = data.drop(columns=data.columns[-1])
            y = data.iloc[:, -1]

        rf = RandomForestClassifier(random_state=42, n_estimators=5)
        gb = GradientBoostingClassifier(random_state=42)
        dt = DecisionTreeClassifier(random_state=42)
        logreg = LogisticRegression(random_state=42, max_iter=1000)

        ensemble_homogeneous = RandomForestClassifier(n_estimators=20, random_state=42)
        ensemble_heterogeneous = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('logreg', logreg)], voting='soft'
        )

        adaboost = AdaBoostClassifier(n_estimators=10, random_state=42)
        easy_ensemble = EasyEnsembleClassifier(n_estimators=20, random_state=42)

        models = {
            'Homogeneous Ensemble': ensemble_homogeneous,
            'Heterogeneous Ensemble': ensemble_heterogeneous,
            'AdaBoost': adaboost,
            'EasyEnsemble': easy_ensemble
        }

        results = []
        result_csv_path = os.path.join(output_dir, f"results_reference_{file}")

        for model_name, model in models.items():
            logging.info(f"Starting evaluation for {model_name}")
            metrics = evaluate_model_with_logging(result_csv_path, model_name, model, X, y)

        logging.info(f"All results saved to {result_csv_path}")
