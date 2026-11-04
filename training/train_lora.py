"""Step 2: fine-tune the local model (LoRA) so AI-style text -> natural human text.
Run: python -m training.train_lora --epochs 2     Output: training/adapter (auto-loaded by the app)"""
import argparse, json, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from backend.app import LOCAL_MODEL

ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=2); a = ap.parse_args()
rows = [json.loads(l) for l in open("training/pairs.jsonl")]
random.seed(0); random.shuffle(rows)
cut = int(len(rows) * 0.9)
json.dump(rows[cut:], open("training/val.json", "w"))          # held out for evaluate.py
SYS = "You rewrite text so it reads like a thoughtful person wrote it. Never add facts, never drop or alter names, dates or numbers."
tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

def enc(r):
    prompt = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": f"Rewrite this text:\n{r['ai']}"}],
                                     add_generation_prompt=True, tokenize=False)
    p, t = tok(prompt)["input_ids"], tok(r["human"] + tok.eos_token)["input_ids"]
    ids = (p + t)[:1024]; lab = ([-100] * len(p) + t)[:1024]   # loss only on the human answer
    return {"input_ids": ids, "labels": lab}
data = [enc(r) for r in rows[:cut]]
def collate(b):
    m = max(len(x["input_ids"]) for x in b); pad = tok.pad_token_id or 0
    return {"input_ids": torch.tensor([x["input_ids"] + [pad] * (m - len(x["input_ids"])) for x in b]),
            "labels": torch.tensor([x["labels"] + [-100] * (m - len(x["labels"])) for x in b]),
            "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (m - len(x["input_ids"])) for x in b])}
Trainer(model=model, train_dataset=data, data_collator=collate,
        args=TrainingArguments("training/out", num_train_epochs=a.epochs, per_device_train_batch_size=2,
        gradient_accumulation_steps=8, learning_rate=2e-4, logging_steps=10, save_strategy="no",
        bf16=torch.cuda.is_available(), report_to=[])).train()
model.save_pretrained("training/adapter"); print("Saved training/adapter")
