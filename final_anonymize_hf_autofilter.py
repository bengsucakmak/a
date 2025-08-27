#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final anonymization with AUTO-FILTER (Ensemble GPU NER + Aho–Corasick replace)

Özellikler:
- Ensemble NER: --model ve opsiyonel --model2 (+ --ensemble {avg,max})
- Skor eşikleri ayarlanabilir (min-conf, min-freq, accept-th, reject-th)
- ALWAYS_INCLUDE + whitelist/blacklist
- Triage dökümü: triage_dump.csv (word, score, ner_conf, freq, bucket)
- Final replace: Aho–Corasick > FlashText > token-set
- NDJSON akışı opsiyonel (--save-ndjson)

Girdi:
  --candidates candidate_list.txt
  --input      all_data.json (JSON array; text: combined_text | yoksa question+answer)
"""

import os, sys, re, json, time, argparse, csv
from typing import List, Dict, Any, Tuple
import pandas as pd
from tqdm.auto import tqdm

from datasets import Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from transformers.pipelines.pt_utils import KeyDataset

# ---- Yer tutucu ve TR yapılandırmaları ----
TEXT_KEY_DEFAULT = "combined_text"
PLACEHOLDER = "[ANONIM_OZEL_AD]"

# Zorunlu dahil (örnek takma adlar)
ALWAYS_INCLUDE = {"ChatDoctor", "Chat Doctor"}

# TR kurum/örgüt ekleri (heuristik)
ORG_SUFFIXES = [
    "a.ş.", "aş", "a.s.", "anonim", "holding", "grup", "şirketi", "ltd", "ltd.", "limited",
    "üniversitesi", "belediyesi", "bakanlığı", "hastanesi", "kliniği", "derneği", "vakfı",
    "enstitüsü", "fakültesi", "okulları", "kurumu"
]

# TR harf duyarlı parçalama (token-set ve frekans için)
TR_SPLIT = re.compile(r"([A-Za-zÇĞİÖŞÜçğıöşü]+)|([^A-Za-zÇĞİÖŞÜçğıöşü]+)", re.UNICODE)

# ---- FlashText (opsiyonel) ----
try:
    from flashtext import KeywordProcessor
    HAS_FLASHTEXT = True
except Exception:
    HAS_FLASHTEXT = False

# ---- Aho–Corasick (tercihli) ----
try:
    import ahocorasick
    HAS_AHO = True
except Exception:
    HAS_AHO = False


# ===================== Yardımcılar =====================

def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def write_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for w in lines:
            f.write(w + "\n")

def unique_keep_order(items: List[str]) -> List[str]:
    seen=set(); out=[]
    for x in items:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def ensure_text_col(df: pd.DataFrame, text_key: str) -> pd.DataFrame:
    if text_key in df.columns:
        return df
    if "question" in df.columns and "answer" in df.columns:
        print("[warn] combined_text yok → question+answer birleştiriliyor...")
        df[text_key] = (df["question"].fillna("") + " " + df["answer"].fillna("")).str.strip()
        return df
    print(f"Error: '{text_key}' yok ve 'question'+'answer' yok.", file=sys.stderr); sys.exit(1)

def looks_clean(w: str) -> bool:
    # yalnızca harf (’ ve ' hariç), sayı yok, uzunluk 2..40
    if not (2 <= len(w) <= 40): return False
    ww = w.replace("’","").replace("'","")
    return ww.isalpha()

def is_titlecase_tr(w: str) -> bool:
    return len(w) > 1 and w[0].isupper()

def has_org_suffix(w: str) -> bool:
    lw = w.lower()
    return any(lw.endswith(suf) for suf in ORG_SUFFIXES)


# ===================== HF NER (GPU) =====================

def init_hf_pipeline(model_name: str, device: int):
    """
    HF token-classification pipeline (PER/ORG), CUDA varsa FP16.
    torch.compile KULLANMIYORUZ (pipeline sınıf kontrolü için).
    """
    import torch
    print(f"[hf] loading: {model_name} | device={device}")
    tok = AutoTokenizer.from_pretrained(model_name)
    kwargs = {}
    if device >= 0 and torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16  # VRAM verimliliği
    mdl = AutoModelForTokenClassification.from_pretrained(model_name, **kwargs)
    if device >= 0 and torch.cuda.is_available():
        mdl = mdl.to(f"cuda:{device}")
    mdl.eval()

    nlp = pipeline(
        "token-classification",
        model=mdl,
        tokenizer=tok,
        device=device,                   # 0=cuda:0, -1=cpu
        aggregation_strategy="simple"
    )
    return nlp

def wrap_for_ner(w: str) -> str:
    # Tek kelimeye minimal bağlam veriyoruz ki NER tetiklensin
    return f"{w} bugün burada."

def ner_confidence(cand: List[str], pipe, batch_size: int) -> Dict[str, float]:
    """
    Aday kelimeler için PER/ORG maksimum skoru (0..1) döndür.
    Datasets + KeyDataset ile GPU üzerinde batched inference.
    """
    clean = [w for w in cand if looks_clean(w)]
    clean = unique_keep_order(clean)

    texts = [wrap_for_ner(w) for w in clean]
    ds = Dataset.from_dict({"text": texts})
    key_ds = KeyDataset(ds, "text")

    conf: Dict[str, float] = {}
    i = 0
    with tqdm(total=len(clean), desc="NER scoring", unit="kelime") as pbar:
        for ents in pipe(key_ds, batch_size=batch_size):
            w = clean[i]
            score = 0.0
            for e in ents:
                lbl = (e.get("entity_group") or e.get("entity") or "").upper()
                if ("PER" in lbl) or ("PERSON" in lbl) or ("ORG" in lbl):
                    score = max(score, float(e.get("score", 0.0)))
            conf[w] = score
            i += 1
            pbar.update(1)
    return conf


# ===================== Dataset içi frekans =====================

def fast_frequency(df: pd.DataFrame, text_key: str, words: List[str], row_batch: int = 100000) -> Dict[str, int]:
    """
    Dataset içinden SADECE verilen adaylar için sıklık sayar.
    TR_SPLIT ile kelimeleme → O(n) tek geçiş.
    """
    lw2orig = {w.lower(): w for w in words}
    freq: Dict[str, int] = {w: 0 for w in words}
    n = len(df)

    with tqdm(total=n, desc="Freq scan", unit="satır") as pbar:
        for i in range(0, n, row_batch):
            sub = df.iloc[i:min(i+row_batch, n)]
            for txt in sub[text_key].astype(str).tolist():
                parts = TR_SPLIT.findall(txt)
                for w, nw in parts:
                    if w:
                        lw = w.lower()
                        if lw in lw2orig:
                            freq[lw2orig[lw]] += 1
            pbar.update(len(sub))
    return freq


# ===================== Heuristik skor =====================

def score_word(w: str, conf: float, freq: int, max_freq: int) -> float:
    """
    Basit birleşik skor [0..1]:
      s = 0.6*conf + 0.15*title + 0.15*org_suffix + 0.1*(min(freq,max)/max)
    """
    title = 1.0 if is_titlecase_tr(w) else 0.0
    orgs  = 1.0 if has_org_suffix(w) else 0.0
    ff = min(freq, max_freq) / max(1.0, float(max_freq))
    return 0.6*conf + 0.15*title + 0.15*orgs + 0.1*ff


# ===================== Replace motorları =====================

def build_aho_automaton(words: List[str]):
    """Case-insensitive Aho–Corasick: kelimeleri lower ile ekler."""
    A = ahocorasick.Automaton()
    for w in words:
        lw = w.lower()
        if lw:
            A.add_word(lw, lw)
    A.make_automaton()
    return A

def _is_alpha(ch: str) -> bool:
    return ch.isalpha()

def _is_word_boundary(s: str, start: int, end: int) -> bool:
    """s[start:end+1] için kelime sınırı (harf olmayan) kontrolü."""
    left_ok  = (start == 0) or (not _is_alpha(s[start-1]))
    right_ok = (end+1 >= len(s)) or (not _is_alpha(s[end+1]))
    return left_ok and right_ok

def replace_with_aho_list(texts: List[str], automaton) -> List[str]:
    """Aho–Corasick ile case-insensitive + kelime-sınırı kontrollü replace."""
    out = []
    for t in texts:
        s = t if isinstance(t, str) else ""
        sl = s.lower()
        matches = []
        for end_i, lw in automaton.iter(sl):
            start_i = end_i - len(lw) + 1
            if _is_word_boundary(s, start_i, end_i):
                matches.append((start_i, end_i))
        if not matches:
            out.append(s); continue
        matches.sort(key=lambda x: (x[0], x[1]))
        merged = []
        last_end = -1
        for a,b in matches:
            if a > last_end:
                merged.append((a,b)); last_end = b
            else:
                if b > last_end:
                    merged[-1] = (merged[-1][0], b); last_end = b
        res = []; prev = 0
        for a,b in merged:
            res.append(s[prev:a]); res.append(PLACEHOLDER); prev = b + 1
        res.append(s[prev:]); out.append("".join(res))
    return out

def replace_with_tokenset_list(texts: List[str], lower_set) -> List[str]:
    """TR kelimeleme + set lookup ile kesin kelime bazlı replace (case-insensitive)."""
    out = []
    for s in texts:
        t = s if isinstance(s, str) else ""
        parts = TR_SPLIT.findall(t)
        chunks = []
        for w, nw in parts:
            if w:
                chunks.append(PLACEHOLDER if w.lower() in lower_set else w)
            else:
                chunks.append(nw)
        out.append("".join(chunks))
    return out


# ===================== Ana Akış =====================

def main():
    ap = argparse.ArgumentParser(description="Auto-filtered final anonymization (Ensemble GPU NER + Aho–Corasick)")
    ap.add_argument("--candidates", default="candidate_list.txt")
    ap.add_argument("--input", default="all_data.json")
    ap.add_argument("--text-key", default=TEXT_KEY_DEFAULT)
    ap.add_argument("--model", default="savasy/bert-base-turkish-ner-cased")
    ap.add_argument("--model2", default="", help="İkinci NER modeli (opsiyonel; ör. WikiANN XLM-R)")
    ap.add_argument("--ensemble", choices=["avg","max"], default="avg", help="Skor birleştirme kuralı")
    ap.add_argument("--device", type=int, default=0)  # 0=cuda:0, -1=cpu
    ap.add_argument("--hf-batch", type=int, default=2048, help="NER batch (VRAM'e göre büyüt)")
    ap.add_argument("--row-batch", type=int, default=100000, help="JSON batch size (final replace & freq)")
    ap.add_argument("--blacklist", default="", help="Asla anonimlenmeyecek kelimeler (opsiyonel)")
    ap.add_argument("--whitelist", default="", help="Daima anonimlenecek kelimeler (opsiyonel)")
    ap.add_argument("--min-conf", type=float, default=0.50, help="Min NER güven skoru (0..1) [daha kapsayıcı]")
    ap.add_argument("--min-freq", type=int, default=1, help="Min dataset frekansı [daha kapsayıcı]")
    ap.add_argument("--max-freq-norm", type=int, default=100, help="Skor için freq normalizasyon üstü")
    ap.add_argument("--accept-th", type=float, default=0.65, help="Otomatik kabul skor eşiği [daha kapsayıcı]")
    ap.add_argument("--reject-th", type=float, default=0.20, help="Otomatik red skor eşiği [daha kapsayıcı]")
    ap.add_argument("--max-review", type=int, default=2000, help="needs_review azami adet (bilgi amaçlı)")
    ap.add_argument("--no-manual", action="store_true", help="Manuel adımı atla (tam otomatik)")
    ap.add_argument("--save-ndjson", action="store_true", help="NDJSON akış çıktısı yaz")
    ap.add_argument("--no-json-array", action="store_true", help="JSON array yazma (sadece NDJSON)")
    args = ap.parse_args()

    # 1) Girdi dosyaları
    cand_path = os.path.abspath(args.candidates)
    data_path = os.path.abspath(args.input)
    if not os.path.exists(cand_path):
        print(f"Error: candidate_list yok: {cand_path}", file=sys.stderr); sys.exit(1)
    if not os.path.exists(data_path):
        print(f"Error: input JSON yok: {data_path}", file=sys.stderr); sys.exit(1)

    # 2) Adaylar + temizleme
    candidates = unique_keep_order(read_lines(cand_path))
    candidates = [w for w in candidates if looks_clean(w)]
    if not candidates:
        print("Uyarı: Aday listesi boş."); sys.exit(0)

    # 3) Kara/Beyaz listeler
    blacklist = set(read_lines(args.blacklist)) if args.blacklist and os.path.exists(args.blacklist) else set()
    whitelist = set(read_lines(args.whitelist)) if args.whitelist and os.path.exists(args.whitelist) else set()

    # 4) NER (GPU) – Ensemble
    pipe1 = init_hf_pipeline(args.model, device=args.device)
    conf1 = ner_confidence(candidates, pipe1, batch_size=args.hf_batch)

    conf = conf1
    if args.model2:
        pipe2 = init_hf_pipeline(args.model2, device=args.device)
        conf2 = ner_confidence(candidates, pipe2, batch_size=args.hf_batch)
        merged = {}
        for w in candidates:
            s1 = float(conf1.get(w, 0.0))
            s2 = float(conf2.get(w, 0.0))
            merged[w] = (s1 + s2)/2.0 if args.ensemble == "avg" else max(s1, s2)
        conf = merged

    # 5) Dataset + text kolonu + frekans
    df = pd.read_json(data_path, orient="records")
    df = ensure_text_col(df, args.text_key)
    freq = fast_frequency(df, args.text_key, candidates, row_batch=args.row_batch)

    # 6) Skorlama ve triage
    scored: List[Tuple[str, float, float, int]] = []  # (word, score, conf, freq)
    for w in candidates:
        c = float(conf.get(w, 0.0))
        f = int(freq.get(w, 0))
        s = score_word(w, c, f, args.max_freq_norm)
        scored.append((w, s, c, f))

    auto_accept = set(whitelist)
    auto_reject = set()
    review_pool: List[Tuple[str, float, float, int]] = []

    for w, s, c, f in scored:
        if w in blacklist:
            auto_reject.add(w); continue
        if w in auto_accept:
            continue
        if c < args.min_conf or f < args.min_freq:
            auto_reject.add(w); continue
        if s >= args.accept_th:
            auto_accept.add(w)
        elif s <= args.reject_th:
            auto_reject.add(w)
        else:
            review_pool.append((w, s, c, f))

    review_pool.sort(key=lambda x: x[1], reverse=True)
    needs_review = [w for (w, s, c, f) in review_pool[:args.max_review]]

    # 6.1 Triage çıktıları
    base_dir = os.path.dirname(cand_path)
    accept_path = os.path.join(base_dir, "auto_accept.txt")
    reject_path = os.path.join(base_dir, "auto_reject.txt")
    review_path = os.path.join(base_dir, "needs_review.txt")
    write_lines(accept_path, sorted(auto_accept))
    write_lines(reject_path, sorted(auto_reject))
    write_lines(review_path, needs_review)

    # 6.2 Triage CSV (debug/kalibrasyon için)
    dump_csv = os.path.join(base_dir, "triage_dump.csv")
    with open(dump_csv, "w", encoding="utf-8", newline="") as fcsv:
        wcsv = csv.writer(fcsv)
        wcsv.writerow(["word","score","ner_conf","freq","bucket"])
        accepted = set(auto_accept)
        rejected = set(auto_reject)
        for w, s, c, f_ in scored:
            if w in accepted: bucket = "accept"
            elif w in rejected: bucket = "reject"
            else: bucket = "review"
            wcsv.writerow([w, f"{s:.4f}", f"{c:.4f}", f_, bucket])

    print(f"[triage] auto_accept: {len(auto_accept)} | auto_reject: {len(auto_reject)} | needs_review: {len(needs_review)}")
    print(f"[debug] triage_dump.csv yazıldı: {dump_csv}")

    # 6.3 ZORUNLU DAHİL: ALWAYS_INCLUDE + whitelist dosyası
    forced = set(ALWAYS_INCLUDE)
    if args.whitelist and os.path.exists(args.whitelist):
        forced |= set(read_lines(args.whitelist))
    if forced:
        before = len(auto_accept)
        auto_accept |= forced
        auto_reject -= forced
        if len(auto_accept) != before:
            print(f"[force] Zorunlu dahil edilenler eklendi: {sorted(forced)}")

    if not args.no_manual and len(needs_review) > 0:
        print(f"[info] İstersen gözden geçir: {review_path} (ilk {args.max_review})")
        input("Devam için ENTER…")

    # 7) Nihai liste (tam otomatik: auto_accept)
    final_list = sorted(auto_accept)
    if not final_list:
        print("Uyarı: final liste boş. Çıkılıyor."); sys.exit(0)

    # 8) Final anonimleştirme motoru
    if HAS_AHO:
        print("[replace] Aho–Corasick kullanılacak (tercihli).")
        aho = build_aho_automaton(final_list)
        engine = ("aho", aho)
    elif HAS_FLASHTEXT:
        print("[replace] FlashText kullanılacak.")
        kp = KeywordProcessor(case_sensitive=False)
        for w in final_list:
            kp.add_keyword(w, PLACEHOLDER)
        engine = ("flash", kp)
    else:
        print("[replace] Aho–Corasick ve FlashText yok → hızlı token-set kullanılacak.")
        final_lower = set(w.lower() for w in final_list)
        engine = ("tokenset", final_lower)

    n = len(df)
    bs = args.row_batch
    ndjson_path = os.path.join(base_dir, "fully_anonymized_data.ndjson")
    ndjson_fp = open(ndjson_path, "w", encoding="utf-8") if args.save_ndjson else None
    out_records = [] if not args.no_json_array else None

    print("[info] Final anonymleştirme başlıyor…")
    t0 = time.time()
    with tqdm(total=n, desc="Final anonymization", unit="kayıt") as pbar:
        for i in range(0, n, bs):
            end = min(i+bs, n)
            sub = df.iloc[i:end].copy()
            texts = sub[args.text_key].astype(str).tolist()

            kind, obj = engine
            if kind == "aho":
                new_txts = replace_with_aho_list(texts, obj)
            elif kind == "flash":
                try:
                    new_txts = [obj.replace_keywords(t) for t in texts]
                except Exception as e:
                    print(f"[warn] FlashText hata: {e} → tokenset'e geçiliyor.")
                    final_lower = set(w.lower() for w in final_list)
                    engine = ("tokenset", final_lower)
                    new_txts = replace_with_tokenset_list(texts, final_lower)
            else:
                new_txts = replace_with_tokenset_list(texts, obj)

            if ndjson_fp is not None:
                for j, txt in enumerate(new_txts):
                    rec = sub.iloc[j].to_dict()
                    rec[args.text_key] = txt
                    ndjson_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if out_records is not None:
                for j, txt in enumerate(new_txts):
                    rec = sub.iloc[j].to_dict()
                    rec[args.text_key] = txt
                    out_records.append(rec)

            pbar.update(end - i)

    if ndjson_fp is not None:
        ndjson_fp.close()
        print(f"[info] Yazıldı: {ndjson_path}")

    if out_records is not None:
        out_json = os.path.join(base_dir, "fully_anonymized_data.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out_records, f, ensure_ascii=False, indent=2)
        print(f"[done] Yazıldı: {out_json}")
    else:
        print("[done] JSON array yazılmadı (--no-json-array).")

    print(f"[perf] Toplam süre: {(time.time()-t0)/60:.1f} dk")

if __name__ == "__main__":
    main()
