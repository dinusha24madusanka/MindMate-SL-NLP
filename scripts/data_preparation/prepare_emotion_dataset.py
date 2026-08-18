from pathlib import Path
import pandas as pd


# =========================
# FILE PATH
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "emotion" / "emotions.csv"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "emotion"
    / "MindMate_Emotion_Model_Ready.csv"
)


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(INPUT_FILE)

print("Original Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# =========================
# KEEP TEXT + LABEL
# =========================

emotion_df = df[
    ["text", "label"]
].copy()


# =========================
# REMOVE MISSING VALUES
# =========================

emotion_df = emotion_df.dropna(
    subset=["text", "label"]
)


# =========================
# REMOVE EXACT DUPLICATES
# =========================

emotion_df = emotion_df.drop_duplicates(
    subset=["text", "label"]
)


# =========================
# LABEL NAMES
# =========================

label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

emotion_df["Emotion_Name"] = (
    emotion_df["label"]
    .map(label_map)
)


# =========================
# ADD SOURCE
# =========================

emotion_df["Source"] = "Kaggle_Emotion"


# =========================
# FINAL COLUMN ORDER
# =========================

emotion_df = emotion_df[
    [
        "text",
        "label",
        "Emotion_Name",
        "Source"
    ]
]


# =========================
# INFORMATION
# =========================

print("\nClean Shape:")
print(emotion_df.shape)

print("\nLabel Distribution:")

print(
    emotion_df["label"]
    .value_counts()
    .sort_index()
)

print("\nEmotion Distribution:")

print(
    emotion_df["Emotion_Name"]
    .value_counts()
)


# =========================
# SAVE
# =========================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

emotion_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


print("\nDONE [OK]")

print(
    "Saved:",
    OUTPUT_FILE
)
