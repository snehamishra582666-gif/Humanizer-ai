"""Step 3: measure real quality on held-out pairs. Run: python -m training.evaluate"""
import json
from backend.app import llm, words, meaning, facts, burstiness, flesch
SYS = "You rewrite text so it reads like a thoughtful person wrote it. Never add facts, never drop or alter names, dates or numbers."
val = json.load(open("training/val.json")); R = {"meaning>=0.85": 0, "facts_all_kept": 0, "length_within_10%": 0, "burstiness_gain": 0}
for r in val:
    o = llm(SYS, f"Rewrite this text:\n{r['ai']}", max_tokens=int(words(r["ai"]) * 2.2) + 40) or ""
    R["meaning>=0.85"] += meaning(r["ai"], o) >= 0.85
    R["facts_all_kept"] += all(x in o for x in facts(r["ai"]))
    R["length_within_10%"] += abs(words(o) - words(r["ai"])) <= 0.1 * words(r["ai"])
    R["burstiness_gain"] += burstiness(o) > burstiness(r["ai"])
print({k: f"{100 * v / len(val):.1f}%" for k, v in R.items()}, f"on {len(val)} held-out passages")
