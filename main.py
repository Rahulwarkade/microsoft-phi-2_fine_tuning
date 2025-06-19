import requests
from bs4 import BeautifulSoup
import json
from transformers import pipeline

# Step 1: Setup
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

urls = [
    "https://toneop.com/",
    "https://toneop.com/about-us",
    "https://toneop.com/blog",
    "https://toneop.com/media",
    "https://toneop.com/faqs",
    "https://toneop.com/careers"
    "https://toneopfit.com/",
    "https://toneopfit.com/about-us",
    "https://toneopfit.com/blogs",
    "https://toneopfit.com/careers",
    "https://toneopfit.com/contact-us",
    "https://toneopfit.com/how-its-work",
    "https://toneopfit.com/shorts",
    "https://toneopfit.com/faq",
    "https://toneopfit.com/privacy-policy",
    "https://toneopfit.com/disclaimer",
    "https://toneopfit.com/terms-and-conditions",
    "https://toneopfit.com/webstories",
    "https://toneopfit.com/transformation-stories",
    "https://toneopfit.com/lab-tests",
    "https://toneopfit.com/lab-tests/advanced-diabetes-checkup-packages",
    "https://toneopfit.com/lab-tests/plus-diabetes-checkup-packages",
    "https://toneopfit.com/lab-tests/essential-diabetes-checkup-packages",
    "https://toneopfit.com/lab-tests/essential-thyroid-profile-test",
    "https://toneopfit.com/lab-tests/advanced-thyroid-profile-test",
    "https://toneopfit.com/lab-tests/plus-thyroid-profile-test",
    "https://toneopfit.com/lab-tests/essential-hypertension-profile-test",
    "https://toneopfit.com/plans",
    "https://toneopfit.com/plans/home-workout-plan",
    "https://toneopfit.com/plans/yoga-plan",
    "https://toneopfit.com/plans/intermittent-fasting-weight-loss",
    "https://toneopfit.com/plans/balanced-diet-weight-loss",
    "https://toneopfit.com/plans/diet-fitness-weight-loss",
    "https://toneopfit.com/plans/360-degree-transformation-plan",
    "https://toneopfit.com/plans/1-year-transformation-plan",
    "https://toneopeats.com/",
    "https://toneopeats.com/menu",
    "https://toneopeats.com/preference",
    "https://toneopeats.com/about-us",
    "https://toneopeats.com/faq",
    "https://toneopeats.com/blogs",
    "https://toneopeats.com/contact-us",
    "https://toneop.care/",
    "https://toneop.care/blogs",
    "https://toneop.care/cart",
    "https://toneop.care/product/all-product",
    "https://toneop.care/product/supplements/combo",
    "https://toneop.care/product/ayurvedic/health-and-fitness-supplements",
    "https://toneop.care/product/nutraceuticals/skincare",
    "https://toneop.care/product/nutraceuticals/superfoods",
    "https://toneop.care/product/nutraceuticals/weight-loss-supplements",
    "https://toneop.care/product/nutraceuticals/power-detox",
    "https://toneop.care/product/nutraceuticals/daily-nutrition/vitamin-360-multivitamins-tablets",
]

def scrape_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        content = []

        for tag in soup.find_all(["h1", "h2", "h3", "p"]):
            text = tag.get_text(strip=True)
            if text:
                content.append(text)

        return "\n".join(content)
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""

# Step 2: Generate Dataset
dataset = []

for url in urls:
    print(f"Scraping: {url}")
    full_text = scrape_text(url)

    if not full_text.strip():
        continue

    # Keep content length under 1024 tokens (approx 1024 * 0.75 = 768 words)
    input_text = full_text[:1500]

    # Summarize content
    try:
        summary = summarizer(input_text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
    except Exception as e:
        print(f"Failed to summarize {url}: {e}")
        summary = "Summary unavailable due to error."

    # Smart instruction
    page_name = url.rstrip("/").split("/")[-1] or "home"
    instruction = f"Summarize the ToneOp {page_name.replace('-', ' ').title()} page"

    dataset.append({
        "instruction": instruction,
        "input": input_text,
        "output": summary
    })

# Step 3: Save as Alpaca-style JSON
with open("toneop_auto_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

from IPython.display import FileLink
FileLink("toneop_auto_dataset.json")


# upload the file to hf
!pip install -q datasets huggingface_hub
from datasets import Dataset
from huggingface_hub import login
import json

# Step 1: Login with your Hugging Face token
login(token="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")  # Replace with your actual token

# Step 2: Load the JSON dataset
with open("toneop_auto_dataset.json", "r") as f:
    data = json.load(f)

# Step 3: Convert to Hugging Face dataset format
hf_dataset = Dataset.from_list(data)

# Step 4: Upload to your Hugging Face Hub
hf_dataset.push_to_hub("imrahulwarkade/phi-2_toneop-dataset")



# Microsoft/Phi-2 LLMs Fine tuning

#📦 Step 1: Install Required Libraries
!pip install transformers==4.40.1 trl==0.8.6 peft==0.11.1 bitsandbytes==0.43.1 accelerate==0.29.2 datasets==2.19.0
# !pip install -q triton==2.0.0 imp
#Step 2: Load Dataset and Model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import torch


# Load your dataset from Hugging Face
dataset = load_dataset("imrahulwarkade/phi-2_toneop-dataset")

# Load tokenizer and model
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

#🧪 Step 4: Prepare Dataset (Alpaca-style)
def format_example(example):
    if example["input"]:
        return f"Instruct: {example['instruction']}\nInput: {example['input']}\nOutput:"
    else:
        return f"Instruct: {example['instruction']}\nOutput:"

dataset = dataset.map(lambda x: {"text": format_example(x) + " " + x["output"]})

#🧩 Step 5: QLoRA Configuration

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "dense", "fc1", "fc2"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)


#⚙️ Step 6: Training Arguments

training_args = TrainingArguments(
    output_dir="./phi2-toneop-qlora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    bf16=False,
    push_to_hub=False,
    report_to="none"
)


# Save the model and verify it 
trainer.model.save_pretrained("phi2-toneop-qlora")
tokenizer.save_pretrained("phi2-toneop-qlora")

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", device_map="auto", load_in_4bit=True, trust_remote_code=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("phi2-toneop-qlora", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "phi2-toneop-qlora")
model.eval()

from transformers import pipeline

pipe = pipeline("text-generation", model="phi2-toneop-qlora", tokenizer=tokenizer, device_map="auto")

prompt = "Instruct: All Products Combo Daily Fitness Skincare Superfoods Weight Loss Detox > > > Vitamin 360 Tablets Vitamin 360.\nOutput:"
result = pipe(prompt, max_new_tokens=100)[0]["generated_text"]
print(result)


# Upload the model to HF

from huggingface_hub import login
login("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")  # paste your token here

from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to the fine-tuned model directory
model_dir = "phi2-toneop-qlora"

# Push tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.push_to_hub("imrahulwarkade/phi2-toneop-finetuned")

# Push model
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.push_to_hub("imrahulwarkade/phi2-toneop-finetuned")


# FastAPI inference server for your fine-tuned Phi-2,
# ✅ Full FastAPI Server for Fine-Tuned Phi-2 (LoRA)

!pip install -q transformers accelerate peft trl bitsandbytes
!pip install -q fastapi uvicorn nest-asyncio pyngrok fastapi-cors

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import uvicorn
import nest_asyncio
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware

# ✅ Apply asyncio patch (for ngrok + uvicorn)
nest_asyncio.apply()

# ✅ Load fine-tuned Phi-2 model
base_model = "microsoft/phi-2"
adapter_repo = "imrahulwarkade/phi2-toneop-finetuned"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float32,   # Use float32 for CPU or float16 for GPU
    device_map="auto",           # Change to "cpu" if no GPU available
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, adapter_repo)

# ✅ Define FastAPI app
app = FastAPI()

# ✅ Enable CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Input format
class Query(BaseModel):
    prompt: str
    max_new_tokens: int = 150

# ✅ Endpoint for generation
@app.post("/generate")
def generate_text(query: Query):
    prompt = query.prompt.strip()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=query.max_new_tokens,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": decoded.strip()}

# ✅ Start ngrok tunnel
!ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN  # Replace once only
public_url = ngrok.connect(8000)
print(f"🚀 Public API URL: {public_url}")

# ✅ Run the FastAPI app
uvicorn.run(app, host="0.0.0.0", port=8000)

#https://xyz-123.ngrok.io/generate
curl -X POST https://<your-ngrok-id>.ngrok.io/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Instruct: List benefits of meditation\nOutput:", "max_new_tokens": 100}'
