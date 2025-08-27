#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unify & Repair JSON Datasets
----------------------------
- Klasördeki tüm .json dosyalarını bulur.
- JSON geçerliliğini dener; tipik format hatalarını onarmaya çalışır.
- Tüm kayıtları tek bir listede birleştirir (question, answer alanlarıyla).
- Yinelenenleri temizler, rapor çıkarır.
- Çıktılar:
    * all_data.json (varsayılan)
    * repaired/<filename>.repaired.json (onarım yapıldıysa)
    * unify_report.txt (özet/uyarılar)
Yalnızca standart kütüphaneler: os, json, glob, re, argparse, hashlib, io, textwrap
"""

import os
import re
import io
import json
import glob
import argparse
import hashlib
from typing import List, Dict, Any, Tuple

TARGET_NAMES = {
    "alpaca_dolly",
    "clean_patient_doctor_qa",
    "cleaned_turkish_agriculture",
    "cleaned_turkish_dolly",
    "cleaned_turkish_medical_qa",
    "cleaned_turkish_random",
    "cleaned_wikipedia_turkish_qa",
    "imdb_turkish_qa_clean",
}

# Hariç tutulacak dosyalar
EXCLUDE_FILES = {
    "processed-qa-dataset_cleaned.json",
}

# ---- Yardımcılar ----

def read_text(path: str) -> str:
    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def strip_bom_and_controls(s: str) -> str:
    if s.startswith("\ufeff"):
        s = s.lstrip("\ufeff")
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", s)
    return s

def remove_trailing_commas(s: str) -> str:
    s = re.sub(r",\s*([\]}])", r"\1", s)
    return s

def detect_ndjson(lines: List[str]) -> bool:
    nonempty = [ln.strip() for ln in lines if ln.strip()]
    if not nonempty:
        return False
    return all((ln.startswith("{") and ln.endswith("}")) for ln in nonempty)

def wrap_bare_objects_as_array(s: str) -> str:
    stripped = s.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return s
    if re.search(r"}\s*,\s*{", stripped):
        return "[" + stripped + "]"
    if re.search(r"}\s*{", stripped) and not stripped.startswith("{") and not stripped.endswith("}"):
        s2 = re.sub(r"}\s*{", "},{", stripped)
        return "[" + s2 + "]"
    return s

def try_parse_as_std_json(text: str) -> Any:
    return json.loads(text)

def try_parse_as_ndjson(text: str) -> List[Any]:
    items = []
    for i, ln in enumerate(text.splitlines(), 1):
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
            items.append(obj)
        except Exception as e:
            raise ValueError(f"NDJSON satırı {i} parse edilemedi: {e}")
    return items

def normalize_to_list(obj: Any) -> List[Any]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, list):
                return v
        return [obj]
    elif isinstance(obj, list):
        return obj
    else:
        return [obj]

CANDIDATE_QUESTION_KEYS = ["question", "prompt", "input", "soru", "q", "query", "instruction"]
CANDIDATE_ANSWER_KEYS   = ["answer", "output", "completion", "yanit", "cevap", "a", "response", "text"]

def pick_field(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
        for kk in d.keys():
            if kk.lower() == k.lower():
                return d[kk]
    return None

def coerce_qa(item: Any) -> Dict[str, str]:
    if not isinstance(item, dict):
        return {"question": str(item), "answer": ""}
    q = pick_field(item, CANDIDATE_QUESTION_KEYS)
    a = pick_field(item, CANDIDATE_ANSWER_KEYS)
    if q is None and a is None:
        return {"question": json.dumps(item, ensure_ascii=False), "answer": ""}
    if q is None:
        q = ""
    if a is None:
        a = ""
    if not isinstance(q, str):
        q = json.dumps(q, ensure_ascii=False)
    if not isinstance(a, str):
        a = json.dumps(a, ensure_ascii=False)
    return {"question": q, "answer": a}

def sha_pair(d: Dict[str, str]) -> str:
    h = hashlib.sha256()
    h.update(("Q:" + d.get("question","")).encode("utf-8"))
    h.update(("A:" + d.get("answer","")).encode("utf-8"))
    return h.hexdigest()

def repair_text_basic(text: str) -> str:
    s = strip_bom_and_controls(text)
    s = remove_trailing_commas(s)
    s = re.sub(r"}\s*{", "},{", s)
    s = wrap_bare_objects_as_array(s)
    return s

def try_full_repair_pipeline(raw: str) -> Tuple[List[Any], str, List[str]]:
    logs = []
    text = raw
    try:
        obj = try_parse_as_std_json(text)
        logs.append("Doğrudan JSON.parse başarılı.")
        return normalize_to_list(obj), text, logs
    except Exception as e1:
        logs.append(f"Doğrudan JSON.parse hata: {e1}")

    lines = text.splitlines()
    if detect_ndjson(lines):
        try:
            arr = try_parse_as_ndjson(text)
            logs.append("NDJSON algılandı ve parse edildi.")
            return normalize_to_list(arr), json.dumps(arr, ensure_ascii=False, indent=2), logs
        except Exception as e2:
            logs.append(f"NDJSON parse de hata: {e2}")

    repaired = repair_text_basic(text)
    if repaired != text:
        try:
            obj = try_parse_as_std_json(repaired)
            logs.append("Hafif onarım sonrası JSON.parse başarılı.")
            return normalize_to_list(obj), repaired, logs
        except Exception as e3:
            logs.append(f"Hafif onarım sonrası hata: {e3}")

    items = []
    salvage_ok = False
    for idx, ln in enumerate(lines, 1):
        s_ln = repair_text_basic(ln.strip())
        if not s_ln:
            continue
        try:
            obj = json.loads(s_ln)
            if isinstance(obj, list):
                items.extend(obj)
            else:
                items.append(obj)
            salvage_ok = True
        except Exception:
            pass

    if salvage_ok and items:
        logs.append("Satır düzeyinde kurtarma uygulandı.")
        repaired_all = json.dumps(items, ensure_ascii=False, indent=2)
        return normalize_to_list(items), repaired_all, logs

    raise ValueError("Otomatik onarım başarısız: JSON biçimi ağır bozuk veya desteklenmeyen tür.")

# ---- Ana akış ----

def main():
    ap = argparse.ArgumentParser(description="Merge & Repair multiple JSON datasets into all_data.json")
    ap.add_argument("--input-dir", required=True, help="JSON dosyalarının bulunduğu klasör")
    ap.add_argument("--output", default="all_data.json", help="Birleşik çıktı dosyası adı")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.input_dir)
    out_path = os.path.abspath(args.output)
    repaired_dir = os.path.join(in_dir, "repaired")
    os.makedirs(repaired_dir, exist_ok=True)

    report_lines = []
    print("[info] Klasör:", in_dir)
    files = sorted(glob.glob(os.path.join(in_dir, "*.json")))
    if not files:
        print("[warn] Klasörde .json dosyası bulunamadı.")
        return

    print(f"[info] {len(files)} adet .json dosyası bulundu. Birleştirme başlıyor...")
    merged: List[Dict[str, str]] = []
    seen = set()

    report_lines.append("Hedef dosya isimleri (bahsedilenler):")
    for name in sorted(TARGET_NAMES):
        report_lines.append(f"  - {name}.json (varsa işlenecek)")

    for path in files:
        fname = os.path.basename(path)

        # --- Hariç tutulacak dosya kontrolü ---
        if fname in EXCLUDE_FILES:
            print(f"[skip] {fname}: hariç tutuldu.")
            report_lines.append(f"[skip] {fname}: hariç tutuldu.")
            continue
        # --------------------------------------

        base, _ = os.path.splitext(fname)
        tagged = " (hedef-listede)" if base in TARGET_NAMES else ""
        print(f"[info] Dosya işleniyor: {fname}{tagged}")

        raw = read_text(path)
        if not raw.strip():
            msg = f"[warn] {fname}: Dosya boş; atlandı."
            print(msg); report_lines.append(msg)
            continue

        try:
            arr, repaired_text, logs = try_full_repair_pipeline(raw)
            if repaired_text.strip() != raw.strip():
                repaired_path = os.path.join(repaired_dir, f"{fname}.repaired.json")
                with io.open(repaired_path, "w", encoding="utf-8") as rf:
                    rf.write(repaired_text)
                report_lines.append(f"[fix] {fname}: Onarıldı → {os.path.relpath(repaired_path, in_dir)}")
            else:
                report_lines.append(f"[ok]  {fname}: Değişiklik gerektirmedi.")

            count_before = len(merged)
            for it in normalize_to_list(arr):
                qa = coerce_qa(it)
                h = sha_pair(qa)
                if h not in seen and (qa["question"] or qa["answer"]):
                    seen.add(h)
                    merged.append(qa)
            count_added = len(merged) - count_before

            report_lines.append(f"      - Kayıt eklendi: {count_added}")
            for lg in logs:
                report_lines.append(f"      * {lg}")

        except Exception as e:
            msg = f"[error] {fname}: Geçerli JSON'a onarılamadı → {e}"
            print(msg)
            report_lines.append(msg)

    print(f"[info] Birleştirme tamamlandı. Toplam kayıt: {len(merged)}")
    with io.open(out_path, "w", encoding="utf-8") as wf:
        json.dump(merged, wf, ensure_ascii=False, indent=2)
    print(f"[info] Yazıldı: {out_path}")

    report_path = os.path.join(in_dir, "unify_report.txt")
    with io.open(report_path, "w", encoding="utf-8") as rf:
        rf.write("\n".join(report_lines))
    print(f"[info] Rapor: {report_path}")

    print("----- Özet -----")
    kept = len(merged)
    print(f"Toplam benzersiz (question, answer): {kept}")
    print("İpuçları:")
    print("- 'repaired/' klasöründe onarılan orijinal dosyaların düz JSON halleri var.")
    print("- 'unify_report.txt' dosyasında her dosya için ayrıntılı logları bulabilirsiniz.")

if __name__ == "__main__":
    main()
