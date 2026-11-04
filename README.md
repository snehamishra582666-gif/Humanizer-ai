# Humanizer (full-stack, fully local, no API key)
1. `pip install -r requirements.txt`
2. `uvicorn backend.app:app --reload` then open http://localhost:8000
The first run downloads three open models once (about 3.5 GB total) and then works offline:
Qwen2.5-3B-Instruct (needs about 8 GB RAM; set HUMANIZER_LOCAL_MODEL=Qwen/Qwen2.5-1.5B-Instruct on weaker machines) (rewriting and generation), MiniLM (meaning score), DistilBERT (sentiment).
Optional: `HUMANIZER_LOCAL_MODEL=Qwen/Qwen2.5-3B-Instruct` for better quality if you have 8 GB+ RAM or a GPU.

## Train on a real dataset (optional, recommended on a GPU / free Colab)
Dataset: Hello-SimpleAI/HC3 (real human answers). Run from the project root:
1. `python -m training.build_pairs --n 3000`  -> training/pairs.jsonl (quality-filtered AI-style -> human pairs)
2. `python -m training.train_lora --epochs 2`  -> training/adapter (the app loads it automatically)
3. `python -m training.evaluate`               -> real meaning / fact / length scores on held-out data
