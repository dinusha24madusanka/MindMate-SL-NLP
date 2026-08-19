from pathlib import Path
import json
import pandas as pd

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)


# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "local"
    / "intent"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# LOAD DATA
# =====================================================

train_df = pd.read_csv(
    DATA_DIR / "intent_train.csv"
)

val_df = pd.read_csv(
    DATA_DIR / "intent_validation.csv"
)

test_df = pd.read_csv(
    DATA_DIR / "intent_test.csv"
)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# =====================================================
# INPUTS
# =====================================================

X_train = train_df["Clean_Text"].astype(str)
y_train = train_df["Intent_ID"].astype(int)

X_val = val_df["Clean_Text"].astype(str)
y_val = val_df["Intent_ID"].astype(int)

X_test = test_df["Clean_Text"].astype(str)
y_test = test_df["Intent_ID"].astype(int)


# =====================================================
# FEATURES
# WORD + CHARACTER TF-IDF
# =====================================================

features = FeatureUnion([
    (
        "word_tfidf",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000,
            sublinear_tf=True
        )
    ),

    (
        "char_tfidf",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=30000,
            sublinear_tf=True
        )
    )
])


# =====================================================
# MODEL
# =====================================================

model = Pipeline([
    (
        "features",
        features
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        )
    )
])


# =====================================================
# TRAIN
# =====================================================

print("\nTraining Intent Baseline...")

model.fit(
    X_train,
    y_train
)

print("Training complete ✅")


# =====================================================
# VALIDATION
# =====================================================

val_pred = model.predict(X_val)

print("\n====================================")
print("VALIDATION RESULTS")
print("====================================")

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
        digits=4,
        zero_division=0
    )
)


# =====================================================
# TEST
# =====================================================

test_pred = model.predict(X_test)

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


print("\n====================================")
print("TEST RESULTS")
print("====================================")

print(
    "Accuracy:",
    round(test_accuracy, 4)
)

print(
    classification_report(
        y_test,
        test_pred,
        digits=4,
        zero_division=0
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_pred
    )
)


# =====================================================
# SAVE METRICS
# =====================================================

metrics = {

    "model":
        "Word + Character TF-IDF + Logistic Regression",

    "task":
        "12-class intent classification",

    "test_accuracy":
        float(test_accuracy),

    "macro_precision":
        float(macro_precision),

    "macro_recall":
        float(macro_recall),

    "macro_f1":
        float(macro_f1),

    "weighted_precision":
        float(weighted_precision),

    "weighted_recall":
        float(weighted_recall),

    "weighted_f1":
        float(weighted_f1)
}


output_file = (
    RESULT_DIR
    / "baseline_intent_metrics.json"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print("\n====================================")
print("INTENT BASELINE COMPLETE")
print("====================================")

print(
    "Metrics saved:",
    output_file
)