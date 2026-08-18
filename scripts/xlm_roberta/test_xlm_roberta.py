import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "FacebookAI/xlm-roberta-base"

print("=" * 50)
print("MindMate-SL XLM-RoBERTa Environment Test")
print("=" * 50)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# LOAD TOKENIZER
# ==========================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded")


# ==========================================
# LOAD MODEL
# Temporary 22-class head for smoke test
# ==========================================

print("\nLoading XLM-RoBERTa model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=22
)

model.to(device)

print("Model loaded")


# ==========================================
# TEST LOCAL-LANGUAGE TEXT
# ==========================================

sample_texts = [
    "heta exam eka nisa mata godak stress",
    "assignment eka complete karaganna ba wage",
    "මට අද ගොඩක් දුකයි",
    "I feel happy today"
]

inputs = tokenizer(
    sample_texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# ==========================================
# FORWARD PASS
# ==========================================

with torch.no_grad():

    outputs = model(
        **inputs
    )

logits = outputs.logits

print("\nOutput shape:", logits.shape)

print(
    "Expected shape:",
    f"({len(sample_texts)}, 22)"
)

print("\nXLM-RoBERTa smoke test complete")