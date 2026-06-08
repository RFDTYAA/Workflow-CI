import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

token = os.getenv("DAGSHUB_TOKEN")
if token:
    dagshub.auth.add_app_token(token)

dagshub.init(
    repo_owner="RFDTYAA",
    repo_name="credit-scoring-mlops",
    mlflow=True
)

mlflow.set_experiment("Credit_Scoring_Advanced_RafiAditya")

with mlflow.start_run(run_name="RandomForest_Advanced_CI"):
    train_df = pd.read_csv("German-Credit-Dataset/german_credit_train_preprocessed.csv")
    test_df = pd.read_csv("German-Credit-Dataset/german_credit_test_preprocessed.csv")

    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
    mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
    mlflow.log_artifact("confusion_matrix.png")

    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-10:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [f"Feature_{i}" for i in indices])
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
    mlflow.log_artifact("feature_importance.png")

    mlflow.sklearn.log_model(best_model, "model")
    print("✅ Training Advanced di CI berhasil!")