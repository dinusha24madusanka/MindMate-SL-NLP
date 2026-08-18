from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================
# FILE PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "emotion"
INPUT_FILE = OUTPUT_DIR / "MindMate_Emotion_Model_Ready.csv"
TRAIN_OUTPUT_FILE = OUTPUT_DIR / "MindMate_Emotion_Train.csv"
VALIDATION_OUTPUT_FILE = OUTPUT_DIR / "MindMate_Emotion_Validation.csv"
TEST_OUTPUT_FILE = OUTPUT_DIR / "MindMate_Emotion_Test.csv"
COMBINED_OUTPUT_FILE = OUTPUT_DIR / "MindMate_Emotion_Final_Split.csv"


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_FILE)

print("Total samples:", len(df))

print("\nOverall distribution:")
print(
    df["Emotion_Name"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ==========================================
# FIRST SPLIT
# 70% TRAIN
# 30% TEMP
# ==========================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)


# ==========================================
# SECOND SPLIT
# TEMP → 15% VALIDATION + 15% TEST
# ==========================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)


# ==========================================
# ADD DATASET SPLIT COLUMN
# ==========================================

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["Dataset_Split"] = "Train"
val_df["Dataset_Split"] = "Validation"
test_df["Dataset_Split"] = "Test"


# ==========================================
# SHOW COUNTS
# ==========================================

print("\n========================")
print("Split Counts")
print("========================")

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ==========================================
# CHECK DISTRIBUTIONS
# ==========================================

for name, data in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:

    print("\n========================")
    print(name)
    print("========================")

    print(
        data["Emotion_Name"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )


# ==========================================
# SAVE FILES
# ==========================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(
    TRAIN_OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

val_df.to_csv(
    VALIDATION_OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

test_df.to_csv(
    TEST_OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ==========================================
# COMBINED FILE
# ==========================================

combined = pd.concat(
    [
        train_df,
        val_df,
        test_df
    ],
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
print(VALIDATION_OUTPUT_FILE)
print(TEST_OUTPUT_FILE)
print(COMBINED_OUTPUT_FILE)
