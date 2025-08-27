#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ULTRA FAST PII MASKER (TR)
- Mode 'hybrid' (default): Regex for structured PII + HF NER (GPU) for PERSON
- Mode 'regex' : Only regex (fastest), skips PERSON NER
- Input: all_data.json (records list). Text column: combined_text
- If missing, combined_text = question + " " + answer
- Outputs:
  * anonymized.ndjson (stream) if --save-ndjson
  * anonymized.json (one big file) if --save-json  [can be heavy]
  * candidate_list.txt (0 < freq <= max_freq), stopwords filtered, + "ChatDoctor", "Chat Doctor"
"""

import os, sys, json, re, argparse, time
from collections import Counter
from typing import List, Dict, Any
import pandas as pd
from tqdm.auto import tqdm
from dateutil import parser as dtparse

TEXT_KEY = "combined_text"
DEFAULT_MAX_FREQ = 500

# ---------- TR stopwords (kısa) ----------
TR_STOP = {
    "ve","veya","ile","de","da","ki","mi","mu","mü","mı","ama","fakat","ancak",
    "bir","bu","şu","o","şöyle","böyle","diye","gibi","çok","az","daha","en",
    "için","üzere","ise","eğer","çünkü","kadar","her","hiç","yok","var",
    "ben","sen","o","biz","siz","onlar","bana","sana","ona","bizi","sizi","onları",
    "benim","senin","onun","bizim","sizin","onların",
    "ne","nasıl","neden","niçin","hangi","nerede","nereden","nereye","zaman",
    "biraz","hemen","zaten","aslında","hala","ya","hem","ya da",
    "dır","dir","dur","dür","tir","tır","tur","tür","m","n","s","y"
}
PLACEHOLDERS = {
    "anonim_isim","anonim_eposta","anonim_telefon","anonim_url",
    "anonim_iban","anonim_tarih","anonim_tc_kimlik_no"
}

# ---------- Regex’ler (hız için derlenmiş) ----------
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_PHONE = re.compile(r"\b(?:\+?90|0)?\s*(?:\d{3}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}|\d{10})\b")
RE_URL   = re.compile(r"\bhttps?://\S+|\bwww\.\S+\b", re.IGNORECASE)
# TR IBAN: TR + 2 kontrol + 5+17 (bank/hesap), toplam 26
RE_IBAN  = re.compile(r"\bTR\d{24}\b", re.IGNORECASE)
# TC Kimlik: 11 haneli, ilk 0 değil (+ checksum doğrulaması fonksiyonu ile)
RE_TCKN  = re.compile(r"\b[1-9]\d{10}\b")
# Tarih (yaklaşık) – çok agresif değil, sadece yaygın biçimler
RE_DATE  = re.compile(r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})\b")

WORD_RE  = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", re.UNICODE)

def is_valid_tckn(s: str) -> bool:
    """Basit checksum doğrulaması (TÜİK algoritması)."""
    if not (len(s) == 11 and s.isdigit() and s[0] != "0"):
        return False
    digits = list(map(int, s))
    d10 = ((sum(digits[0:9:2]) * 7) - sum(digits[1:8:2])) % 10
    d11 = (sum(digits[:10]) % 10)
    return (digits[9] == d10) and (digits[10] == d11)

def mask_structured(text: str) -> str:
    if not text:
        return text
    # E-posta
    text = RE_EMAIL.sub("[ANONIM_EPOSTA]", text)
    # URL
    text = RE_URL.sub("[ANONIM_URL]", text)
    # IBAN
    text = RE_IBAN.sub("[ANONIM_IBAN]", text)
    # Telefon (çok geniş; URL/e-posta sonrası çalıştırıyoruz)
    text = RE_PHONE.sub("[ANONIM_TELEFON]", text)
    # Tarih – bulununca doğrulamak için dateutil deneyelim (çok kısa durumlardan kaçınmak için try/except)
    def _date_repl(m):
        s = m.group(0)
        try:
            dtparse.parse(s, dayfirst=True, fuzzy=False)
            return "[ANONIM_TARIH]"
        except Exception:
            return s
    text = RE_DATE.sub(_date_repl, text)
    # TCKN – regex + checksum
    def _tckn_repl(m):
        s = m.group(0)
        return "[ANONIM_TC_KIMLIK_NO]" if is_valid_tckn(s) else s
    text = RE_TCKN.sub(_tckn_repl, text)
    return text

# ---------- HF NER (sadece PERSON) ----------
HF_PIPE = None
def init_hf_person_pipeline(model_name: str, device: int = 0, batch_size: int = 32, fp16: bool = True):
    """HuggingFace token-classification pipeline (PERSON only)."""
    global HF_PIPE
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForTokenClassification.from_pretrained(model_name)
    HF_PIPE = pipeline(
        "token-classification",
        model=mdl,
        tokenizer=tok,
        device=device,         # 0: cuda:0, -1: cpu
        aggregation_strategy="simple",
        batch_size=batch_size,
        torch_dtype="auto" if fp16 else None
    )

def mask_person_with_hf(text: str) -> str:
    """Model sadece PERSON etiketlerini döndürsün; onları [ANONIM_ISIM] ile değiştir."""
    if not text or HF_PIPE is None:
        return text
    try:
        ents = HF_PIPE(text, truncation=True)
    except Exception:
        return text
    # bazı TR NER modelleri etiketleri 'B-PER/I-PER' yerine 'PERSON' olarak verir; normalize edelim
    spans = []
    for e in ents:
        lbl = (e.get("entity_group") or e.get("entity") or "").upper()
        if "PER" in lbl or "PERSON" in lbl:
            spans.append((e["start"], e["end"]))
    if not spans:
        return text
    # çakışmayı önlemek için sondan başa değiştir
    s = text
    for a,b in sorted(spans, key=lambda x: x[0], reverse=True):
        s = s[:a] + "[ANONIM_ISIM]" + s[b:]
    return s

def tokens_from_text(s: str):
    if not isinstance(s, str):
        return
    for m in WORD_RE.finditer(s.lower()):
        yield m.group(0)

def process_batch(texts: List[str], mode: str) -> List[str]:
    out = []
    if mode == "regex":
        for t in texts:
            out.append(mask_structured(t))
        return out
    # hybrid
    for t in texts:
        t2 = mask_structured(t)
        t3 = mask_person_with_hf(t2)
        out.append(t3)
    return out

def main():
    ap = argparse.ArgumentParser(description="Ultra Fast TR PII (regex / hybrid)")
    ap.add_argument("--input", required=True, help="all_data.json (records list)")
    ap.add_argument("--text-key", default=TEXT_KEY, help="text column name (default: combined_text)")
    ap.add_argument("--mode", choices=["regex","hybrid"], default="hybrid", help="regex-only or hybrid")
    ap.add_argument("--model", default="savasy/bert-base-turkish-ner-cased",
                    help="HF NER model (only used in hybrid). Örn: savasy/bert-base-turkish-ner-cased")
    ap.add_argument("--device", type=int, default=0, help="GPU id (hybrid). -1 for CPU")
    ap.add_argument("--batch-size", type=int, default=20000, help="records per loop (I/O/freq batching)")
    ap.add_argument("--hf-batch", type=int, default=64, help="HF pipeline batch size (hybrid)")
    ap.add_argument("--max-freq", type=int, default=DEFAULT_MAX_FREQ, help="candidate upper freq threshold (<=)")
    ap.add_argument("--save-ndjson", action="store_true", help="write anonymized.ndjson (stream)")
    ap.add_argument("--save-json", action="store_true", help="write anonymized.json (RAM heavy)")
    args = ap.parse_args()

    path = os.path.abspath(args.input)
    if not os.path.exists(path):
        print(f"Error: input not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] loading: {path}")
    df = pd.read_json(path, orient="records")

    # combined_text üret
    if args.text_key not in df.columns:
        if "question" in df.columns and "answer" in df.columns:
            print("[warn] combined_text yok → question+answer birleştiriliyor...")
            df[args.text_key] = (df["question"].fillna("") + " " + df["answer"].fillna("")).str.strip()
        else:
            print(f"Error: '{args.text_key}' yok ve 'question'+'answer' yok.", file=sys.stderr)
            sys.exit(1)

    texts = df[args.text_key].fillna("").tolist()
    n = len(texts)

    # Hybrid ise HF pipeline başlat
    if args.mode == "hybrid":
        print(f"[hf] init model: {args.model} | device={args.device} | batch={args.hf_batch}")
        init_hf_person_pipeline(args.model, device=args.device, batch_size=args.hf_batch, fp16=True)

    ndjson_fp = open(os.path.join(os.path.dirname(path), "anonymized.ndjson"), "w", encoding="utf-8") if args.save_ndjson else None
    json_buf: List[Dict[str, Any]] = [] if args.save_json else None

    freqs = Counter()
    bs = args.batch_size
    t0 = time.time()

    with tqdm(total=n, desc=f"Mode={args.mode}", unit="kayıt") as pbar:
        for i in range(0, n, bs):
            batch = texts[i:i+bs]
            out_batch = process_batch(batch, mode=args.mode)

            # frekans
            for txt in out_batch:
                for w in WORD_RE.findall(txt.lower()):
                    if w not in TR_STOP and w not in PLACEHOLDERS and len(w) > 1:
                        freqs[w] += 1

            # akış yazımı
            if ndjson_fp is not None:
                for j, txt in enumerate(out_batch):
                    rec = df.iloc[i + j].to_dict()
                    rec[args.text_key] = txt
                    ndjson_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if json_buf is not None:
                for j, txt in enumerate(out_batch):
                    rec = df.iloc[i + j].to_dict()
                    rec[args.text_key] = txt
                    json_buf.append(rec)

            pbar.update(len(batch))

    if ndjson_fp is not None:
        ndjson_fp.close()
        print("[info] anonymized.ndjson yazıldı")

    if json_buf is not None:
        bigp = os.path.join(os.path.dirname(path), "anonymized.json")
        with open(bigp, "w", encoding="utf-8") as f:
            json.dump(json_buf, f, ensure_ascii=False, indent=2)
        print(f"[info] {bigp} yazıldı")

    # candidate list
    candidates = [w for w,c in freqs.items() if 0 < c <= args.max_freq]
    candidates.sort(key=lambda w: freqs[w], reverse=True)
    out_list = os.path.join(os.path.dirname(path), "candidate_list.txt")
    with open(out_list, "w", encoding="utf-8") as f:
        for w in candidates:
            f.write(w + "\n")
        f.write("ChatDoctor\n")
        f.write("Chat Doctor\n")

    dt = time.time() - t0
    print(f"[done] {n} kayıt | süre: {dt/60:.1f} dk | mod: {args.mode}")
    print(f"[info] candidate_list.txt yazıldı: {out_list}")

if __name__ == "__main__":
    main()
