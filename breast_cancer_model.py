import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

df.to_csv("breast_cancer_dataset.csv", index=False)

print("Dataset saved as breast_cancer_dataset.csv")
print("Dataset Shape:", df.shape)
print(df.head())

dataset = pd.read_csv("breast_cancer_dataset.csv")

X = dataset.drop("target", axis=1)
y = dataset["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


baseline_model = RandomForestClassifier(random_state=42)
baseline_model.fit(X_train, y_train)

y_pred_baseline = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(y_test, y_pred_baseline)

print("\n================ BASELINE MODEL ================")
print("Accuracy:", baseline_accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred_baseline))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_baseline))


param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_grid_model = grid_search.best_estimator_
y_pred_grid = best_grid_model.predict(X_test)
grid_accuracy = accuracy_score(y_test, y_pred_grid)

print("\n================ GRID SEARCH ================")
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", grid_accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred_grid))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_grid))


param_dist = {
    "n_estimators": np.arange(50, 300, 50),
    "max_depth": [None] + list(np.arange(5, 30, 5)),
    "min_samples_split": np.arange(2, 15)
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

best_random_model = random_search.best_estimator_
y_pred_random = best_random_model.predict(X_test)
random_accuracy = accuracy_score(y_test, y_pred_random)

print("\n================ RANDOM SEARCH ================")
print("Best Parameters:", random_search.best_params_)
print("Accuracy:", random_accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred_random))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_random))

print("\n=================================================")
print("FINAL MODEL COMPARISON")
print("=================================================")
print("Baseline Accuracy      :", baseline_accuracy)
print("Grid Search Accuracy   :", grid_accuracy)
print("Random Search Accuracy :", random_accuracy)
print("=================================================")
