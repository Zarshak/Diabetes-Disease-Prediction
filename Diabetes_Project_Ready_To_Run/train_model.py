from pathlib import Path
import urllib.request
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
DATA_FILE = DATA_DIR / "diabetes.csv"

# Public mirror of the Pima Indians Diabetes dataset.
DATA_URL = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"

if not DATA_FILE.exists():
    print("Dataset not found locally. Downloading it now...")
    try:
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    except Exception as exc:
        raise SystemExit(
            "Could not download the dataset. Connect to the internet and run again, "
            "or place diabetes.csv inside the data folder. Details: " + str(exc)
        )

df = pd.read_csv(DATA_FILE)
if "Outcome" not in df.columns and "y" in df.columns:
    df = df.rename(columns={"y": "Outcome"})
required = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Dataset columns are missing: {missing}")

df = df[required].copy()
for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["Outcome"])
X = df.drop(columns=["Outcome"])
y = df["Outcome"].astype(int)

# In this dataset, zero is not a realistic measurement for several medical fields;
# treat those values as missing and impute using the median learned from training data.
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
X[zero_as_missing] = X[zero_as_missing].replace(0, np.nan)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

algorithms = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7)
}

results = []
trained = {}
for name, estimator in algorithms.items():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0)
    }
    results.append(metrics)
    trained[name] = pipe
    print("\n" + name)
    print(classification_report(y_test, pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

# Choose the model with the highest F1-score on the held-out test split.
best = max(results, key=lambda item: item["F1-score"])
best_name = best["Model"]
joblib.dump(trained[best_name], MODEL_DIR / "diabetes_model.pkl")
(ROOT / "models" / "features.json").write_text(
    __import__("json").dumps(list(X.columns), indent=2), encoding="utf-8"
)
pd.DataFrame(results).to_csv(ROOT / "model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(pd.DataFrame(results).to_string(index=False))
print(f"\nSaved selected model: {best_name}")
print("Saved model to models/diabetes_model.pkl")
print("Saved comparison to model_comparison.csv")
