from pathlib import Path
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =====================================================
# PROJECT PATH
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOCAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "local"
    / "model_ready"
    / "MindMate_Model_Ready_Balanced_Split.xlsx"
)


# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_excel(
    LOCAL_FILE,
    sheet_name="Model_Ready_Local"
)

print("Total samples:", len(df))


# =====================================================
# TRAIN / VALIDATION / TEST
# =====================================================

train_df = df[
    df["Dataset_Split_v2"] == "Train"
].copy()

val_df = df[
    df["Dataset_Split_v2"] == "Validation"
].copy()

test_df = df[
    df["Dataset_Split_v2"] == "Test"
].copy()


print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# =====================================================
# INPUTS
# =====================================================

X_train = train_df["Clean_Text"].astype(str)
y_train = train_df["Label_Code"]

X_val = val_df["Clean_Text"].astype(str)
y_val = val_df["Label_Code"]

X_test = test_df["Clean_Text"].astype(str)
y_test = test_df["Label_Code"]


# =====================================================
# BASELINE
# TF-IDF + LOGISTIC REGRESSION
# =====================================================

model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=30000
            )
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# =====================================================
# TRAIN
# =====================================================

print("\nTraining baseline model...")

model.fit(
    X_train,
    y_train
)

print("Training complete")


# =====================================================
# VALIDATION
# =====================================================

val_pred = model.predict(X_val)

print("\n================================")
print("VALIDATION RESULTS")
print("================================")

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_val,
            val_pred
        ),
        4
    )
)

print(
    classification_report(
        y_val,
        val_pred,
        digits=4
    )
)


# =====================================================
# TEST
# =====================================================

test_pred = model.predict(X_test)

print("\n================================")
print("TEST RESULTS")
print("================================")

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            test_pred
        ),
        4
    )
)

print(
    classification_report(
        y_test,
        test_pred,
        digits=4
    )
)


# =====================================================
# CONFUSION MATRIX
# =====================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_pred
    )
)


print("\nBASELINE MODEL COMPLETE")


from sklearn.metrics import precision_recall_fscore_support
import json


# =====================================================
# SAVE BASELINE METRICS
# =====================================================

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


test_accuracy = accuracy_score(
    y_test,
    test_pred
)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )
)

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        y_test,
        test_pred,
        average="weighted",
        zero_division=0
    )
)


metrics = {
    "model": "TF-IDF + Logistic Regression",
    "test_accuracy": float(test_accuracy),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "weighted_precision": float(weighted_precision),
    "weighted_recall": float(weighted_recall),
    "weighted_f1": float(weighted_f1)
}


with open(
    RESULT_DIR / "baseline_local_metrics.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print(
    "\nMetrics saved to:",
    RESULT_DIR / "baseline_local_metrics.json"
)
