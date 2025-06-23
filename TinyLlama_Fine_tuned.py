!pip install -q transformers==4.40.1 trl==0.8.6 peft==0.11.1 bitsandbytes==0.43.1 accelerate==0.29.2 datasets==2.19.0
!pip install -U datasets fsspec
!pip install -U triton==2.0.0

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch

# ✅ Step 1: Load dataset
dataset = load_dataset("imrahulwarkade/toneopbot-tinyllama-dataset", split="train")

# ✅ Step 2: Load model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ✅ Step 3: Tokenize dataset (with labels)
def tokenize(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])


# ✅ Step 4: QLoRA + 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ✅ Step 5: LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ✅ Step 6: Load model and prepare for training
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# ✅ Step 7: Training arguments
training_args = TrainingArguments(
    output_dir="./tinyllama-toneopbot",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True, 
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    optim="paged_adamw_8bit"
)

# ✅ Step 8: SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset,
    peft_config=lora_config,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=1024,
    packing=True
)
# ⚠️ Temporary fix for floating_point_ops bug
trainer.floating_point_ops = lambda *args, **kwargs: 0

# ✅ Step 9: Start training
trainer.train()

# ✅ Step 10: Save LoRA adapter
trainer.model.save_pretrained("tinyllama-toneopbot-lora")
tokenizer.save_pretrained("tinyllama-toneopbot-lora")

print("✅ Training complete. LoRA adapter saved to: tinyllama-toneopbot-lora")
