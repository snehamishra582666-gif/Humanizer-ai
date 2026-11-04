"""AI Text Humanizer API: extract -> (generate) -> condense/expand -> humanize -> quality gate -> history."""
import io, os, re, json, sqlite3, statistics, time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

DB = os.path.join(os.path.dirname(__file__), "history.db")
MODEL = os.getenv("HUMANIZER_MODEL", "claude-sonnet-5-5")
app = FastAPI(title="AI Text Humanizer")

def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY, created REAL, source TEXT,
      output TEXT, in_words INT, out_words INT, metrics TEXT)""")
    return c

# ---------- text extraction (txt / md / pdf / docx) ----------
def extract(name: str, data: bytes) -> str:
    ext = name.lower().rsplit(".", 1)[-1]
    if ext in ("txt", "md", "text"): return data.decode("utf-8", "ignore")
    if ext == "pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    if ext == "docx":
        import docx
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    raise HTTPException(400, f"Unsupported file type: .{ext}. Use TXT, MD, PDF or DOCX.")

# ---------- NLP helpers ----------
words = lambda t: len(re.findall(r"\b[\w'’-]+\b", t))
sents = lambda t: [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]
_models = {}

def sentiment(sentence_list):
    """Per-sentence sentiment with DistilBERT (deep learning); neutral fallback if unavailable."""
    try:
        if "sent" not in _models:
            from transformers import pipeline
            _models["sent"] = pipeline("sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english")
        return [r["label"].lower() for r in _models["sent"]([s[:400] for s in sentence_list])]
    except Exception:
        return ["neutral"] * len(sentence_list)

def meaning(a, b):
    """Cosine similarity of sentence embeddings (MiniLM); word-overlap fallback."""
    try:
        if "emb" not in _models:
            from sentence_transformers import SentenceTransformer
            _models["emb"] = SentenceTransformer("all-MiniLM-L6-v2")
        from sentence_transformers import util
        e = _models["emb"].encode([a, b], convert_to_tensor=True)
        return float(util.cos_sim(e[0], e[1]))
    except Exception:
        A, B = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
        return len(A & B) / max(1, len(A | B)) ** 0.9

def facts(t):  # numbers, dates, and capitalised names must survive
    return set(re.findall(r"\b\d[\d,.%/-]*\b", t)) | set(re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b", t))

def flesch(t):
    w, s = max(1, words(t)), max(1, len(sents(t)))
    syl = sum(max(1, len(re.findall(r"[aeiouy]+", x.lower()))) for x in re.findall(r"[A-Za-z]+", t))
    return round(206.835 - 1.015 * w / s - 84.6 * syl / w, 1)

def burstiness(t):
    L = [words(s) for s in sents(t)]
    return round(statistics.pstdev(L), 1) if len(L) > 1 else 0.0

CLICHES = {r"\bdelve into\b": "dig into", r"\bin today's fast-paced world\b": "these days",
 r"\bit is important to note that\b": "note that", r"\bmoreover,?\b": "also,", r"\bfurthermore,?\b": "and",
 r"\butili[sz]e\b": "use", r"\bleverage\b": "use", r"\bin conclusion,?\b": "overall,",
 r"\bplays a crucial role\b": "matters a lot", r"\bnavigate the complexities\b": "deal with it",
 r"\bit is\b": "it's", r"\bdo not\b": "don't", r"\bcannot\b": "can't", r"\bthey are\b": "they're"}
def declich(t):
    for p, r in CLICHES.items(): t = re.sub(p, r, t, flags=re.I)
    return t

# ---------- LLM (runs locally, no API key) ----------
LOCAL_MODEL = os.getenv("HUMANIZER_LOCAL_MODEL", "Qwen/Qwen2.5-3B-Instruct")
def llm(system, user, max_tokens=700, temp=0.85):
    try:
        if "lm" not in _models:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
            lm = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL, torch_dtype="auto", device_map="auto")
            ad = os.getenv("HUMANIZER_ADAPTER", os.path.join(os.path.dirname(__file__), "..", "training", "adapter"))
            cfg = os.path.join(ad, "adapter_config.json")
            if os.path.isfile(cfg):  # fine-tuned LoRA from training/train_lora.py
                base = json.load(open(cfg)).get("base_model_name_or_path")
                if base == LOCAL_MODEL:
                    from peft import PeftModel
                    lm = PeftModel.from_pretrained(lm, ad); print("Loaded fine-tuned adapter")
                else:
                    print(f"Skipping adapter: it was trained on {base}, but the app uses {LOCAL_MODEL}. Retrain or switch model.")
            _models["lm"] = (tok, lm)
        tok, lm = _models["lm"]
        ids = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                      add_generation_prompt=True, return_tensors="pt").to(lm.device)
        out = lm.generate(ids, max_new_tokens=max_tokens, do_sample=True, temperature=temp, top_p=0.9)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    except Exception as e:
        print("Local model unavailable:", e); return None

# ---------- AI-style estimate (local GPT-2 perplexity + burstiness) ----------
def ppl_sents(sl):
    """Perplexity per sentence with DistilGPT-2. Very predictable (low) text reads as machine-written."""
    try:
        import torch
        if "gpt" not in _models:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            _models["gpt"] = (AutoTokenizer.from_pretrained("distilgpt2"), AutoModelForCausalLM.from_pretrained("distilgpt2").eval())
        tok, g = _models["gpt"]; res = []
        for x in sl:
            ids = tok(x, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
            if ids.shape[1] < 3: res.append(60.0); continue
            with torch.no_grad(): res.append(float(torch.exp(g(ids, labels=ids).loss)))
        return res
    except Exception:
        return [60.0] * len(sl)

def ai_likeness(t):
    """0-100 rough estimate of how machine-like the style is (lower = more natural). Not a real detector."""
    sl = sents(t)
    if not sl: return 0
    p = ppl_sents(sl); mp = sum(p) / len(p)
    hits = sum(len(re.findall(c, t, re.I)) for c in list(CLICHES)[:8])
    return round(min(100, 70 / (1 + (mp / 45) ** 1.5) + 30 / (1 + burstiness(t) / 6) + 5 * hits))

SYS2 = "You rewrite one sentence so it sounds like a real person wrote it. Output only the sentence."
def polish(out, must, rounds=2):
    """Re-write only the most predictable sentences, accept a change only if meaning and facts survive."""
    for _ in range(rounds):
        sl = sents(out); p = ppl_sents(sl); changed = False
        for i in [i for i, x in enumerate(p) if x < 40 and words(sl[i]) > 6][:8]:
            new = llm(SYS2, "Rewrite this sentence with a different structure and plainer, more specific wording. "
                      "Keep all names, dates, numbers and the meaning. Vary the length.\n" + sl[i], max_tokens=120, temp=1.0)
            if not new: continue
            new = declich(new.strip().split("\n")[0])
            if meaning(sl[i], new) >= 0.8 and all(f in new for f in must if f in sl[i]) and ppl_sents([new])[0] > p[i]:
                sl[i] = new; changed = True
        out = " ".join(sl)
        if not changed: break
    return out

def chunks(t, size=180):
    out, cur = [], []
    for s_ in sents(t):
        cur.append(s_)
        if words(" ".join(cur)) >= size: out.append(" ".join(cur)); cur = []
    return out + ([" ".join(cur)] if cur else [])

STRENGTH = {"light": "Make small edits only: swap stiff phrases, keep structure.",
 "medium": "Vary sentence length (mix short and long), vary openers, use contractions and active voice.",
 "heavy": "Rewrite freely for a natural voice: strong rhythm variety, concrete words, an occasional aside or question."}

class Req(BaseModel):
    text: str = ""
    prompt: str = ""
    strength: str = "medium"
    tone: str = "neutral"
    target_words: int = 0

def run(req: Req):
    src = req.text.strip()
    if not src and not req.prompt.strip(): raise HTTPException(400, "Add some text, upload a file, or write a prompt.")
    steps = []
    if not src:  # Generate mode
        src = llm("You are a writer. Write clear, factual first drafts.", req.prompt) \
              or req.prompt
        steps.append("generated")
    src_sent = sents(src); src_sent_lab = sentiment(src_sent)
    tgt = req.target_words or words(src)
    must = sorted(facts(src))
    best, repair, attempts = None, "", 0
    for attempts in range(1, 4):
        sys = ("You rewrite text so it reads like a thoughtful person wrote it. Never add facts, never drop or alter "
               "names, dates or numbers, keep each sentence's sentiment. Output only the rewritten text. "
               f"Tone: {req.tone}. {STRENGTH.get(req.strength, STRENGTH['medium'])} "
               "Mix very short and long sentences, avoid repeating sentence openers, prefer concrete everyday words over abstract ones. Avoid cliches like 'delve into', 'moreover', 'in today's fast-paced world'.")
        parts = []
        for ch in chunks(src):
            ct = max(5, round(words(ch) * tgt / max(1, words(src))))
            usr = f"Target length: about {ct} words.\n{repair}\nRewrite this text:\n{ch}"
            parts.append(llm(sys, usr, max_tokens=int(ct * 2.2) + 60) or ch)
        out = declich(" ".join(parts))
        w, ms = words(out), meaning(src, out)
        lost = [f for f in must if f not in out]
        out_lab = sentiment(sents(out)); same = sum(a == b for a, b in zip(src_sent_lab, out_lab))
        sent_keep = same / max(1, min(len(src_sent_lab), len(out_lab)))
        ok_len = abs(w - tgt) <= 0.05 * tgt
        m = {"in_words": words(src), "out_words": w, "target_words": tgt, "length_ok": ok_len,
             "meaning": round(ms, 3), "facts_kept": round(1 - len(lost) / max(1, len(must)), 3),
             "dropped_facts": lost, "sentiment_kept": round(sent_keep, 3), "readability": flesch(out),
             "burstiness": burstiness(out), "attempts": attempts}
        score = (ok_len, not lost, ms)
        if best is None or score > best[0]: best = (score, out, m)
        if ok_len and not lost and ms >= 0.85: break
        repair = ("FIX LAST ATTEMPT: " + (f"it had {w} words, need {tgt}. " if not ok_len else "")
                  + (f"restore these: {', '.join(lost)}. " if lost else "")
                  + ("stay closer to the original meaning." if ms < 0.85 else ""))
    _, out, m = best
    m["ai_before"] = ai_likeness(out)
    out = polish(out, must)
    w2, lost2 = words(out), [f for f in must if f not in out]
    m.update(ai_after=ai_likeness(out), out_words=w2, length_ok=abs(w2 - tgt) <= 0.05 * tgt, meaning=round(meaning(src, out), 3),
             dropped_facts=lost2, facts_kept=round(1 - len(lost2) / max(1, len(must)), 3), readability=flesch(out), burstiness=burstiness(out))
    m["llm"] = "lm" in _models
    with db() as c:
        cur = c.execute("INSERT INTO runs(created,source,output,in_words,out_words,metrics) VALUES(?,?,?,?,?,?)",
                        (time.time(), src, out, m["in_words"], m["out_words"], json.dumps(m)))
    return {"id": cur.lastrowid, "source": src, "output": out, "metrics": m,
            "sentences": [{"text": s, "sentiment": l} for s, l in zip(src_sent, src_sent_lab)]}

@app.post("/api/humanize")
def humanize(req: Req): return run(req)

@app.post("/api/humanize-file")
async def humanize_file(file: UploadFile = File(...), strength: str = Form("medium"),
                        tone: str = Form("neutral"), target_words: int = Form(0)):
    text = extract(file.filename, await file.read())
    if not text.strip(): raise HTTPException(400, "No readable text found in that file (scanned PDFs need OCR).")
    return run(Req(text=text, strength=strength, tone=tone, target_words=target_words))

@app.get("/api/history")
def history():
    return [dict(r) | {"metrics": json.loads(r["metrics"])} for r in
            db().execute("SELECT * FROM runs ORDER BY id DESC LIMIT 100")]

@app.delete("/api/history/{rid}")
def delete(rid: int):
    with db() as c: c.execute("DELETE FROM runs WHERE id=?", (rid,))
    return {"ok": True}

class Pdf(BaseModel): text: str; title: str = "Humanized text"
@app.post("/api/pdf")
def pdf(p: Pdf):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from xml.sax.saxutils import escape
    buf, st = io.BytesIO(), getSampleStyleSheet()
    flow = [Paragraph(escape(p.title), st["Title"]), Spacer(1, 12)] + \
           [x for para in p.text.split("\n\n") for x in (Paragraph(escape(para).replace("\n", "<br/>"), st["BodyText"]), Spacer(1, 8))]
    SimpleDocTemplate(buf, pagesize=A4).build(flow); buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf",
                             headers={"Content-Disposition": 'attachment; filename="humanized.pdf"'})

app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend"), html=True))
