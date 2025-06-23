import json
import random
from datasets import load_dataset

# System prompt used in all training messages
system_prompt = "You are a helpful assistant."

def format_tinyllama_chat(messages):
    """Convert list of messages to a single string for TinyLlama training."""
    formatted_chat = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        if role == "system":
            formatted_chat.append(f"<|system|>\n{content}</s>")
        elif role == "user":
            formatted_chat.append(f"<|user|>\n{content}</s>")
        elif role == "assistant":
            formatted_chat.append(f"<|assistant|>\n{content}</s>")
    return {"text": "\n".join(formatted_chat)}

# 🔹 Load your scraped ToneOp dataset
with open("toneop_alpaca_style.json", "r", encoding="utf-8") as f:
    toneop_alpaca = json.load(f)

# 🔹 Convert ToneOp dataset
toneop_tinyllama = []
for row in toneop_alpaca:
    instruction = row["instruction"].strip()
    input_text = row["input"].strip()
    output = row["output"].strip()
    user_prompt = f"{instruction}\n{input_text}" if input_text else instruction
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": output}
    ]
    toneop_tinyllama.append(format_tinyllama_chat(messages))

# 🔹 Load tatsu-lab/alpaca dataset from HF
alpaca = load_dataset("tatsu-lab/alpaca", split="train")

# 🔹 Convert Alpaca dataset
alpaca_tinyllama = []
for row in alpaca:
    instruction = row["instruction"].strip()
    input_text = row["input"].strip()
    output = row["output"].strip()
    user_prompt = f"{instruction}\n{input_text}" if input_text else instruction
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": output}
    ]
    alpaca_tinyllama.append(format_tinyllama_chat(messages))

# 🔹 Merge and shuffle
merged_dataset = toneop_tinyllama + alpaca_tinyllama
random.shuffle(merged_dataset)

# 🔹 Save as final JSONL file for TinyLlama SFT
output_path = "toneopbot_tinyllama_dataset.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for item in merged_dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Final TinyLlama-compatible dataset saved as: {output_path}")
