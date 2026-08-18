from pathlib import Path

import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "local"
    / "cleaned"
    / "MindMate_SL_Research_Data(20260815-120805).xlsx"
)
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "local"
    / "model_ready"
    / "MindMate_Model_Ready_Balanced_Split.xlsx"
)

# =========================
# 1. LOAD DATA
# =========================

df = pd.read_excel(
    INPUT_FILE,
    sheet_name="Model_Ready_Local"
)

print("Total samples:", len(df))
print("Participants:", df["Participant_ID"].nunique())


# =========================
# 2. REMOVE OLD SPLIT ONLY
#    FROM NEW WORKING COPY
# =========================

df["Dataset_Split_v2"] = ""


# =========================
# 3. FIRST SPLIT
#    Train+Validation / Test
# =========================

X = df["Clean_Text"].astype(str)
y = df["Label_Code"].astype(str)
groups = df["Participant_ID"].astype(str)

sgkf_test = StratifiedGroupKFold(
    n_splits=7,
    shuffle=True,
    random_state=42
)

train_val_idx, test_idx = next(
    sgkf_test.split(
        X,
        y,
        groups
    )
)

train_val_df = df.iloc[train_val_idx].copy()
test_df = df.iloc[test_idx].copy()


# =========================
# 4. SECOND SPLIT
#    Train / Validation
# =========================

X_tv = train_val_df["Clean_Text"].astype(str)
y_tv = train_val_df["Label_Code"].astype(str)
groups_tv = train_val_df["Participant_ID"].astype(str)

sgkf_val = StratifiedGroupKFold(
    n_splits=6,
    shuffle=True,
    random_state=43
)

train_rel_idx, val_rel_idx = next(
    sgkf_val.split(
        X_tv,
        y_tv,
        groups_tv
    )
)

train_df = train_val_df.iloc[train_rel_idx].copy()
val_df = train_val_df.iloc[val_rel_idx].copy()


# =========================
# 5. ASSIGN SPLIT
# =========================

df.loc[train_df.index, "Dataset_Split_v2"] = "Train"
df.loc[val_df.index, "Dataset_Split_v2"] = "Validation"
df.loc[test_df.index, "Dataset_Split_v2"] = "Test"


# =========================
# 6. VERIFY PARTICIPANTS
# =========================

train_participants = set(
    df[df["Dataset_Split_v2"] == "Train"]["Participant_ID"]
)

val_participants = set(
    df[df["Dataset_Split_v2"] == "Validation"]["Participant_ID"]
)

test_participants = set(
    df[df["Dataset_Split_v2"] == "Test"]["Participant_ID"]
)

print("\nParticipant overlap:")

print(
    "Train vs Validation:",
    len(train_participants & val_participants)
)

print(
    "Train vs Test:",
    len(train_participants & test_participants)
)

print(
    "Validation vs Test:",
    len(val_participants & test_participants)
)


# =========================
# 7. SHOW SPLIT COUNTS
# =========================

print("\nSplit counts:")
print(
    df["Dataset_Split_v2"].value_counts()
)


# =========================
# 8. LABEL DISTRIBUTION
# =========================

for split in [
    "Train",
    "Validation",
    "Test"
]:

    print(
        "\n======================"
    )

    print(split)

    temp = df[
        df["Dataset_Split_v2"] == split
    ]

    distribution = (
        temp["Label_Code"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print(
        distribution.round(2)
    )


# =========================
# 9. SAVE
# =========================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_excel(
    OUTPUT_FILE,
    index=False,
    sheet_name="Model_Ready_Local"
)

print(
    "\nSaved:",
    OUTPUT_FILE
)
