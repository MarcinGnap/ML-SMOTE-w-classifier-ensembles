import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from imblearn.ensemble import EasyEnsembleClassifier
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def evaluate_model_with_logging(model, X_train, X_test, y_train, y_test, model_name):
    logging.info(f"Training model: {model_name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

    logging.info(f"{model_name} Evaluation Metrics:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-score: {f1:.4f}")
    if roc_auc is not None:
        logging.info(f"ROC AUC: {roc_auc:.4f}")

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'ROC AUC': roc_auc
    }


if __name__ == '__main__':
    data = pd.read_csv('./data/telecom_churn.csv')

    X = data.drop(columns=['Churn'])
    y = data['Churn']

    rf = RandomForestClassifier(random_state=42, n_estimators=2)
    gb = GradientBoostingClassifier(random_state=42)
    dt = DecisionTreeClassifier(random_state=42)
    logreg = LogisticRegression(random_state=42, solver='saga', max_iter=1000)

    ensemble_homogeneous = RandomForestClassifier(n_estimators=20, random_state=42)
    ensemble_heterogeneous = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('logreg', logreg)], voting='soft'
    )

    adaboost = AdaBoostClassifier(random_state=42)
    easy_ensemble = EasyEnsembleClassifier(random_state=42)

    models = {
        'Random Forest': rf,
        'Gradient Boosting': gb,
        'Decision Tree': dt,
        'Logistic Regression': logreg,
        'Homogeneous Ensemble': ensemble_homogeneous,
        'Heterogeneous Ensemble': ensemble_heterogeneous,
        'AdaBoost': adaboost,
        'EasyEnsemble': easy_ensemble
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = []

    for model_name, model in models.items():
        logging.info(f"Starting evaluation for {model_name}")
        metrics = evaluate_model_with_logging(model, X_train, X_test, y_train, y_test, model_name)
        results.append({'Model': model_name, **metrics})

    results_df = pd.DataFrame(results)
    results_df.to_csv('./data/result_reference.csv', index=False)

    logging.info("All results saved to 'result_reference.csv'")
