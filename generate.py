from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "./model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# IMPORTANT for GPT-2
tokenizer.pad_token = tokenizer.eos_token
model.eval()

while True:
    prompt = input("\nEnter prompt (or type 'exit'): ")
    if prompt.lower() == "exit":
        break

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )

    print("\nGenerated text:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))