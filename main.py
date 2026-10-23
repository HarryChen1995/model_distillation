import os
import subprocess


def select_gpu_by_name(target_name: str):
    # Must run before torch initializes CUDA. Pin CUDA's ordering to nvidia-smi's (PCI bus)
    # and expose only the matching GPU, so Trainer doesn't wrap the model in DataParallel.
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


os.environ["TOKENIZERS_PARALLELISM"] = "false"
select_gpu_by_name("p100")

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)



class CausalKnowledgeDistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        
        # Isolate and freeze the teacher model
        if self.teacher_model is not None:
            self.teacher_model.eval()
            self.teacher_model.to(self.args.device)
            for param in self.teacher_model.parameters():
                param.requires_grad = False
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. Forward pass on the Student
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        student_logits = outputs_student.logits

        # 2. Forward pass on the Teacher 
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
            teacher_logits = outputs_teacher.logits

        # 3. Compute KL Divergence Loss over token predictions
        # Causal model logits are shape: [batch_size, sequence_length, vocab_size]
        loss_function_kd = nn.KLDivLoss(reduction="batchmean")
        
        soft_loss = loss_function_kd(
            torch.log_softmax(student_logits / self.temperature, dim=-1),
            torch.softmax(teacher_logits / self.temperature, dim=-1)
        ) * (self.temperature ** 2)

        # 4. Joint Loss: balance student language loss and teacher mimicry
        loss = self.alpha * student_loss + (1.0 - self.alpha) * soft_loss
        return (loss, outputs_student) if return_outputs else loss
        
        


def main():
    print(f"--- Running Raw Text Distillation Loop on: {torch.cuda.get_device_name(0)} ---")

    teacher_id = "EleutherAI/pythia-160m"
    student_id = "EleutherAI/pythia-70m"

    print("Loading models and tokenizers...")
    tokenizer = AutoTokenizer.from_pretrained(student_id)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token for causal text generation

    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_id, dtype=torch.float32)
    student_model = AutoModelForCausalLM.from_pretrained(student_id, dtype=torch.float32)

    # 4. Stream and Process Raw Open Text Data
    print("Streaming fineweb-edu dataset from Hugging Face...")
    raw_dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    # Labels are built by the collator (mlm=False) after padding, with pad positions set to -100
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    # Slicing 2,000 text document samples
    dataset_slice = raw_dataset.take(2000)
    tokenized_dataset = dataset_slice.map(tokenize_function, batched=True, remove_columns=["text", "id", "url", "score"])
    
    # Convert stream to a structural list for standard trainer compatibility
    train_list = list(tokenized_dataset)
    train_dataset = train_list[:1800]
    eval_dataset = train_list[1800:]

    # Causal language data collator to handle dynamic token padding on the fly
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. Define Training Arguments Optimized for P100 (16GB VRAM)
    training_args = TrainingArguments(
        output_dir="./p100_text_distillation",
        num_train_epochs=1,
        per_device_train_batch_size=4,   # Small batch to keep GPU power and heat down
        per_device_eval_batch_size=4,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        learning_rate=5e-5,
        weight_decay=0.01,
        
        # Core flags for P100 hardware configuration:
        fp16=True,                      # Standard 16-bit floating operations
        dataloader_num_workers=0,
        report_to="none"
    )

    # 6. Initialize Custom Trainer
    trainer = CausalKnowledgeDistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        alpha=0.5,                      # Equal weight on text data labels and teacher guidance
        temperature=2.0
    )

    # 7. Start Training
    print("Launching text training loop...")
    trainer.train()
    print("Distillation Complete! Saving model...")
    trainer.save_model("./final_distilled_text_model")

if __name__ == "__main__":
    main()