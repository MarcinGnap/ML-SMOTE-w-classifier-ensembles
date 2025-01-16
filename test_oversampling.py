import logging
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTEN, SVMSMOTE, BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def test_balancing(modifier, models, X, y, use_modifier=True):
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

            if use_modifier and modifier:
                X_train, y_train = modifier.fit_resample(X_train, y_train)

            model.fit(X_train, y_train)
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

    models = {
        'Random Forest': rf,
        'Gradient Boosting': gb,
        'Decision Tree': dt,
        'Logistic Regression': logreg,
        'Homogeneous Ensemble': ensemble_homogeneous,
        'Heterogeneous Ensemble': ensemble_heterogeneous
    }

    modifiers = {
        "No Modifier": None,
        "SMOTE": SMOTE(random_state=42),
        "SMOTEN": SMOTEN(random_state=42),
        "SVMSMOTE": SVMSMOTE(random_state=42),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=42)
    }

    end_results = {}

    logging.info("Testing without modifiers (baseline)...")
    end_results["No Modifier"] = test_balancing(None, models, X, y, use_modifier=False)

    for mod_name, modifier in modifiers.items():
        if mod_name != "No Modifier":
            logging.info(f"Testing with modifier: {mod_name}")
            end_results[mod_name] = test_balancing(modifier, models, X, y, use_modifier=True)

    logging.info("\n" + "=" * 20 + " Summary " + "=" * 20)
    csv_results = []

    for name, part_dict in end_results.items():
        logging.info(f"\n{name}")
        for model_name, metrics in part_dict.items():
            logging.info(f"Model: {model_name}")
            logging.info(
                f"Accuracy: {metrics['Accuracy']:.4f}; "
                f"Precision: {metrics['Precision']:.4f}; "
                f"Recall: {metrics['Recall']:.4f}; "
                f"F1-score: {metrics['F1-score']:.4f}"
            )
            if metrics['ROC AUC'] is not None:
                logging.info(f"ROC AUC: {metrics['ROC AUC']:.4f}")

            csv_results.append({
                "Modifier": name,
                "Model": model_name,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1-score": metrics["F1-score"],
                "ROC AUC": metrics["ROC AUC"]
            })

    results_df = pd.DataFrame(csv_results)
    results_df.to_csv('./results_w_and_wo_modifiers.csv', index=False)

    logging.info("Summary saved to 'results_w_and_wo_modifiers.csv'")
