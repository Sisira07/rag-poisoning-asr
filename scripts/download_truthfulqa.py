"""
Create data/raw/truthfulqa/truthfulqa.jsonl (question, best_answer, correct_answers, incorrect_answers).

Tries the current Hugging Face copy first, then falls back to the original CSV on GitHub.
    python scripts/download_truthfulqa.py [--out data/raw/truthfulqa/truthfulqa.jsonl]
"""
import argparse
import json
from pathlib import Path


def from_huggingface():
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    return [{"question": x["question"], "best_answer": x["best_answer"],
             "correct_answers": x["correct_answers"], "incorrect_answers": x["incorrect_answers"]} for x in ds]


def from_github_csv():
    import pandas as pd
    url = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
    df = pd.read_csv(url)
    split = lambda s: [a.strip() for a in str(s).split(";") if a.strip()]
    return [{"question": r["Question"], "best_answer": r["Best Answer"],
             "correct_answers": split(r["Correct Answers"]),
             "incorrect_answers": split(r["Incorrect Answers"])} for _, r in df.iterrows()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/truthfulqa/truthfulqa.jsonl")
    out = Path(ap.parse_args().out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = None
    try:
        rows = from_huggingface()
        print("Loaded from Hugging Face:", len(rows))
    except Exception as e:  # noqa: BLE001
        print("Hugging Face load failed:", repr(e)[:300])
    if rows is None:
        rows = from_github_csv()
        print("Loaded from GitHub CSV:", len(rows))
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Saved", out)


if __name__ == "__main__":
    main()
