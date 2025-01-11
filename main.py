import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

data = pd.read_csv('./data/creditcard.csv')

X = data.drop(columns=['Class'])
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# definicje klasyfikatorów
# na razie wyłączyłem SVC bo dla takiej liczby rekordów się nie wykonuje najlepiej
# dałem też LR na saga bo jest problem z liczbą iteracji na domyślnej
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
dt = DecisionTreeClassifier(random_state=42)
logreg = LogisticRegression(random_state=42, solver='saga', max_iter=1000)
# svm = SVC(probability=True, random_state=42)

# homogeniczne i heterogeniczne
ensemble_homogeneous = RandomForestClassifier(n_estimators=100, random_state=42)
ensemble_heterogeneous = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('logreg', logreg)], voting='soft'
)

models = {
    'Random Forest': rf,
    'Gradient Boosting': gb,
    'Decision Tree': dt,
    'Logistic Regression': logreg,
    # 'SVM': svm,
    'Homogeneous Ensemble': ensemble_homogeneous,
    'Heterogeneous Ensemble': ensemble_heterogeneous
}

results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"\nModel: {name}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]) if hasattr(model, 'predict_proba') else None
    if auc:
        print(f"ROC AUC: {auc:.4f}")

    results[name] = classification_report(y_test, y_pred, output_dict=True)

print("\nPodsumowanie wyników:")
for name, result in results.items():
    print(f"Model: {name}")
    print(f"Precision: {result['1']['precision']:.4f}")
    print(f"Recall: {result['1']['recall']:.4f}")
    print(f"F1-score: {result['1']['f1-score']:.4f}")
    print("-")
