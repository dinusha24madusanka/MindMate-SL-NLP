from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOCAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "local"
    / "model_ready"
    / "MindMate_Model_Ready_Balanced_Split.xlsx"
)

STRESS_TRAIN = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stress"
    / "MindMate_Stress_Train.csv"
)

STRESS_TEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stress"
    / "MindMate_Stress_Test.csv"
)

EMOTION_TRAIN = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "emotion"
    / "MindMate_Emotion_Train.csv"
)

EMOTION_VAL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "emotion"
    / "MindMate_Emotion_Validation.csv"
)

EMOTION_TEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "emotion"
    / "MindMate_Emotion_Test.csv"
)


# =====================================================
# 1. LOCAL MINDMATE DATASET
# =====================================================

print("\n======================================")
print("1. LOCAL MINDMATE DATASET")
print("======================================")

local_df = pd.read_excel(
    LOCAL_FILE,
    sheet_name="Model_Ready_Local"
)

print("Total samples:", len(local_df))

print("\nLabel distribution:")
print(
    local_df["Label_Code"]
    .value_counts()
    .sort_index()
)

print("\nSplit counts:")
print(
    local_df["Dataset_Split_v2"]
    .value_counts()
)

print(
    "\nMissing Clean_Text:",
    local_df["Clean_Text"].isna().sum()
)

print(
    "Missing Label_Code:",
    local_df["Label_Code"].isna().sum()
)


# Participant leakage check

train_people = set(
    local_df[
        local_df["Dataset_Split_v2"] == "Train"
    ]["Participant_ID"]
)

val_people = set(
    local_df[
        local_df["Dataset_Split_v2"] == "Validation"
    ]["Participant_ID"]
)

test_people = set(
    local_df[
        local_df["Dataset_Split_v2"] == "Test"
    ]["Participant_ID"]
)

print("\nParticipant overlap:")

print(
    "Train vs Validation:",
    len(train_people & val_people)
)

print(
    "Train vs Test:",
    len(train_people & test_people)
)

print(
    "Validation vs Test:",
    len(val_people & test_people)
)


# =====================================================
# 2. STRESS DATASET
# =====================================================

print("\n======================================")
print("2. STRESS DATASET")
print("======================================")

stress_train = pd.read_csv(
    STRESS_TRAIN
)

stress_test = pd.read_csv(
    STRESS_TEST
)

print("Train samples:", len(stress_train))
print("Test samples:", len(stress_test))

print("\nTrain labels:")
print(
    stress_train["label"]
    .value_counts()
    .sort_index()
)

print("\nTest labels:")
print(
    stress_test["label"]
    .value_counts()
    .sort_index()
)

print(
    "\nMissing Train text:",
    stress_train["text"].isna().sum()
)

print(
    "Missing Test text:",
    stress_test["text"].isna().sum()
)


# =====================================================
# 3. EMOTION DATASET
# =====================================================

print("\n======================================")
print("3. EMOTION DATASET")
print("======================================")

emotion_train = pd.read_csv(
    EMOTION_TRAIN
)

emotion_val = pd.read_csv(
    EMOTION_VAL
)

emotion_test = pd.read_csv(
    EMOTION_TEST
)

print("Train samples:", len(emotion_train))
print("Validation samples:", len(emotion_val))
print("Test samples:", len(emotion_test))


for name, df in [
    ("Train", emotion_train),
    ("Validation", emotion_val),
    ("Test", emotion_test)
]:

    print(f"\n{name} labels:")

    print(
        df["Emotion_Name"]
        .value_counts()
    )

    print(
        f"{name} missing text:",
        df["text"].isna().sum()
    )


# =====================================================
# FINISHED
# =====================================================

print("\n======================================")
print("FINAL DATASET CHECK COMPLETE [OK]")
print("======================================")
