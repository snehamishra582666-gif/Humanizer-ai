"""Step 1: build (AI-style -> human) training pairs from real human writing (HC3 dataset).
Human text is the TARGET, so the model learns real human style. The AI-style input is produced from the
human text, so meaning always matches. Pairs failing meaning/fact checks are dropped.
Run: python -m training.build_pairs --n 3000   (GPU or Colab strongly recommended)"""
import argparse, json, random
from datasets import load_dataset
from backend.app import llm, words, meaning, facts

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3000); a = ap.parse_args()
ds = load_dataset("Hello-SimpleAI/HC3", "all", trust_remote_code=True)["train"]
human = [t for r in ds for t in r["human_answers"] if 60 <= words(t) <= 220]
random.seed(0); random.shuffle(human)
SYS = "You rewrite text in the polished, formal, generic style of an AI assistant. Keep every fact, name and number. Output only the text."
kept = 0
with open("training/pairs.jsonl", "w") as f:
    for h in human:
        if kept >= a.n: break
        ai = llm(SYS, h, max_tokens=int(words(h) * 2.2) + 40)
        if not ai or meaning(h, ai) < 0.8 or any(x not in ai for x in facts(h)): continue
        f.write(json.dumps({"ai": ai, "human": h}) + "\n"); kept += 1
        if kept % 50 == 0: print(kept, "pairs")
print("done:", kept)
