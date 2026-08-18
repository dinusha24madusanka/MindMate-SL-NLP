from pathlib import Path
import pandas as pd


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


# =====================================================
# SCENARIO / INTENT INVENTORY
# =====================================================

intent_summary = (
    df.groupby(
        ["Scenario_ID", "Scenario_Name"]
    )
    .size()
    .reset_index(name="Sample_Count")
    .sort_values("Scenario_ID")
)


print("\n========================================")
print("MINDMATE-SL SCENARIO / INTENT INVENTORY")
print("========================================")

print(
    intent_summary.to_string(
        index=False
    )
)

print(
    "\nTotal scenario classes:",
    intent_summary["Scenario_ID"].nunique()
)

print(
    "Total samples:",
    intent_summary["Sample_Count"].sum()
)


# =====================================================
# SPLIT DISTRIBUTION PER SCENARIO
# =====================================================

split_table = pd.crosstab(
    index=[
        df["Scenario_ID"],
        df["Scenario_Name"]
    ],
    columns=df["Dataset_Split_v2"]
)

print("\n========================================")
print("SCENARIO COUNTS BY SPLIT")
print("========================================")

print(split_table)


# =====================================================
# SAVE RESULT
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

output_file = (
    RESULT_DIR
    / "intent_label_inventory.csv"
)

intent_summary.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)

print(
    "\nSaved:",
    output_file
)

print(
    "\nINTENT LABEL INSPECTION COMPLETE"
)