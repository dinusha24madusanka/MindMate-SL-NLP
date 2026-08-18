from pathlib import Path

import pandas as pd

# =========================
# FILE NAMES
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "stress"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "stress"

TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"
TRAIN_OUTPUT_FILE = OUTPUT_DIR / "MindMate_Stress_Train.csv"
TEST_OUTPUT_FILE = OUTPUT_DIR / "MindMate_Stress_Test.csv"
COMBINED_OUTPUT_FILE = OUTPUT_DIR / "MindMate_Stress_Model_Ready.csv"

# =========================
# LOAD DATA
# =========================

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Original Train Shape:", train_df.shape)
print("Original Test Shape:", test_df.shape)

print("\nTrain columns:")
print(train_df.columns.tolist())

# =========================
# KEEP ONLY TEXT + LABEL
# =========================

train_clean = train_df[
    ["text", "label"]
].copy()

test_clean = test_df[
    ["text", "label"]
].copy()

# =========================
# REMOVE MISSING ROWS
# =========================

train_clean = train_clean.dropna(
    subset=["text", "label"]
)

test_clean = test_clean.dropna(
    subset=["text", "label"]
)

# =========================
# REMOVE EXACT DUPLICATES
# =========================

train_clean = train_clean.drop_duplicates(
    subset=["text", "label"]
)

test_clean = test_clean.drop_duplicates(
    subset=["text", "label"]
)

# =========================
# ADD SOURCE + SPLIT
# =========================

train_clean["Dataset_Split"] = "Train"
test_clean["Dataset_Split"] = "Test"

train_clean["Source"] = "Dreaddit"
test_clean["Source"] = "Dreaddit"

# =========================
# FINAL COLUMN ORDER
# =========================

columns = [
    "text",
    "label",
    "Dataset_Split",
    "Source"
]

train_clean = train_clean[columns]
test_clean = test_clean[columns]

# =========================
# DISPLAY INFORMATION
# =========================

print("\nClean Train Shape:")
print(train_clean.shape)

print("\nClean Test Shape:")
print(test_clean.shape)

print("\nTrain Label Distribution:")
print(
    train_clean["label"]
    .value_counts()
    .sort_index()
)

print("\nTest Label Distribution:")
print(
    test_clean["label"]
    .value_counts()
    .sort_index()
)

# =========================
# SAVE SEPARATE FILES
# =========================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_clean.to_csv(
    TRAIN_OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

test_clean.to_csv(
    TEST_OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

# =========================
# COMBINED FILE
# =========================

combined = pd.concat(
    [train_clean, test_clean],
    ignore_index=True
)

combined.to_csv(
    COMBINED_OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\nDONE [OK]")
print("Saved:")
print(TRAIN_OUTPUT_FILE)
print(TEST_OUTPUT_FILE)
print(COMBINED_OUTPUT_FILE)
