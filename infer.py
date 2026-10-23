import os
import subprocess


def select_gpu_by_name(target_name: str):
    # Must run before torch initializes CUDA. Pin CUDA's ordering to nvidia-smi's (PCI bus)
    # and expose only the matching GPU.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return
    for line in out.strip().splitlines():
        index, name = (part.strip() for part in line.split(",", 1))
        if target_name.lower() in name.lower():
            os.environ["CUDA_VISIBLE_DEVICES"] = index
            return
    print(f"Device containing '{target_name}' not found. Using default CUDA devices.")


select_gpu_by_name("p100")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = "./final_distilled_text_model"
# main.py doesn't pass the tokenizer to the Trainer, so it isn't saved with the model
TOKENIZER_ID = "EleutherAI/pythia-70m"
PROMPT = "The water cycle is"
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7  # 0 = greedy decoding


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Loading {MODEL_DIR} on {device_name}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).to(device)
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=TEMPERATURE > 0,
            temperature=TEMPERATURE if TEMPERATURE > 0 else None,
            top_p=0.9 if TEMPERATURE > 0 else None,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
