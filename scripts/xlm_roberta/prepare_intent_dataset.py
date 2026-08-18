from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "local"
    / "model_ready"
    / "MindMate_Model_Ready_Balanced_Split.xlsx"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "local"
    / "intent"
)

CONFIG_DIR = PROJECT_ROOT / "config"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CONFIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# SCENARIO → INTENT MAPPING
# ==========================================

scenario_to_intent = {

    1:  (0, "ACADEMIC_STRESS"),
    6:  (0, "ACADEMIC_STRESS"),

    2:  (1, "CAMPUS_SAFETY_INJUSTICE"),
    3:  (1, "CAMPUS_SAFETY_INJUSTICE"),

    4:  (2, "FINANCIAL_STRESS"),

    5:  (3, "DAILY_CAMPUS_LOGISTICS"),
    18: (3, "DAILY_CAMPUS_LOGISTICS"),

    7:  (4, "SOCIAL_ISOLATION"),

    8:  (5, "ACADEMIC_SUCCESS"),
    9:  (5, "ACADEMIC_SUCCESS"),

    10: (6, "FINAL_PAPER_RELIEF"),
    11: (6, "FINAL_PAPER_RELIEF"),

    12: (7, "SOCIAL_EVENT"),
    13: (7, "SOCIAL_EVENT"),

    14: (8, "PERSONAL_WIN"),
    15: (8, "PERSONAL_WIN"),

    16: (9, "LECTURE_MANAGEMENT"),
    17: (9, "LECTURE_MANAGEMENT"),

    19: (10, "CAMPUS_RESOURCE_REQUEST"),
    20: (10, "CAMPUS_RESOURCE_REQUEST"),

    21: (11, "FOOD_AND_CANTEEN"),
    22: (11, "FOOD_AND_CANTEEN")
}


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_excel(
    INPUT_FILE,
    sheet_name="Model_Ready_Local"
)

print("Original samples:", len(df))


# ==========================================
# MAP INTENTS
# ==========================================

df["Intent_ID"] = df["Scenario_ID"].map(
    lambda x: scenario_to_intent[int(x)][0]
)

df["Intent_Name"] = df["Scenario_ID"].map(
    lambda x: scenario_to_intent[int(x)][1]
)


# ==========================================
# FINAL COLUMNS
# ==========================================

intent_df = df[
    [
        "Sample_ID",
        "Participant_ID",
        "Scenario_ID",
        "Scenario_Name",
        "Clean_Text",
        "Intent_ID",
        "Intent_Name",
        "Script_Type",
        "Dataset_Split_v2"
    ]
].copy()

intent_df = intent_df.rename(
    columns={
        "Dataset_Split_v2": "Dataset_Split"
    }
)


# ==========================================
# VERIFY
# ==========================================

print("\nIntent Distribution:")

print(
    intent_df["Intent_Name"]
    .value_counts()
)

print(
    "\nMissing Intent IDs:",
    intent_df["Intent_ID"].isna().sum()
)


# ==========================================
# SPLIT FILES
# ==========================================

train_df = intent_df[
    intent_df["Dataset_Split"] == "Train"
].copy()

val_df = intent_df[
    intent_df["Dataset_Split"] == "Validation"
].copy()

test_df = intent_df[
    intent_df["Dataset_Split"] == "Test"
].copy()


print("\nTrain:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ==========================================
# SAVE
# ==========================================

train_df.to_csv(
    OUTPUT_DIR / "intent_train.csv",
    index=False,
    encoding="utf-8"
)

val_df.to_csv(
    OUTPUT_DIR / "intent_validation.csv",
    index=False,
    encoding="utf-8"
)

test_df.to_csv(
    OUTPUT_DIR / "intent_test.csv",
    index=False,
    encoding="utf-8"
)

intent_df.to_csv(
    OUTPUT_DIR / "intent_all.csv",
    index=False,
    encoding="utf-8"
)


# ==========================================
# SAVE LABEL MAP
# ==========================================

label_map = {
    str(intent_id): intent_name
    for intent_id, intent_name
    in sorted(
        set(scenario_to_intent.values())
    )
}

with open(
    CONFIG_DIR / "intent_label_map.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        label_map,
        file,
        indent=4
    )


print("\nSaved:")
print("intent_train.csv")
print("intent_validation.csv")
print("intent_test.csv")
print("intent_all.csv")
print("intent_label_map.json")

print("\nINTENT DATASET PREPARATION COMPLETE")