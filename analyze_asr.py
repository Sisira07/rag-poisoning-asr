"""
analyze_asr.py -- turn data/asr/asr_results.jsonl into tables + a chart.

Usage:
    python analyze_asr.py                       # heuristic labels
    python analyze_asr.py --labels j            # LLM-judge labels (needs --judge in the run)
    python analyze_asr.py --in path/to/file.jsonl --outdir data/asr
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt(k, n):
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{100*k/n:.0f}% ({k}/{n}) [{100*lo:.0f}-{100*hi:.0f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/asr/asr_results.jsonl")
    ap.add_argument("--outdir", default="data/asr")
    ap.add_argument("--labels", choices=["h", "j"], default="h")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    clean = {r["qid"]: r for r in recs if r["strategy"] == "clean"}
    strategies = sorted({r["strategy"] for r in recs if r["strategy"] != "clean"})
    conds = ["A", "B", "G"] + [c for c in ("R", "R2", "R3") if any(f"label_{c}_h" in r for r in recs)]
    names = {"A": "No defense", "B": "Agent filter", "G": "Agent gate", "R": "Filter+lexical ref", "R2": "Filter+NLI ref (top-K)", "R3": "Filter+NLI ref (corpus)"}

    def lab(r, c):
        return r.get(f"label_{c}_{a.labels}")

    rows, chart = [], defaultdict(dict)
    lines = [f"Labels: {'heuristic' if a.labels == 'h' else 'LLM judge'}  |  95% Wilson CI in []\n"]

    for s in strategies:
        rs = [r for r in recs if r["strategy"] == s]
        att = [r for r in rs if r["attacked"]]
        non = [r for r in rs if not r["attacked"]]
        lines.append(f"### Attack: {s}   (attacked={len(att)}, untouched={len(non)})")
        lines.append("| Condition | ASR (attacked) | Mixed (attacked) | Blocked (attacked) | Correct (attacked) "
                     "| Correct (untouched) | Blocked (untouched) |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in conds:
            asr = sum(lab(r, c) == "false" for r in att)
            mix = sum(lab(r, c) == "mixed" for r in att)
            blk = sum(lab(r, c) == "blocked" for r in att)
            cor = sum(lab(r, c) == "correct" for r in att)
            cor_n = sum(lab(r, c) == "correct" for r in non)
            blk_n = sum(lab(r, c) == "blocked" for r in non)
            lines.append(f"| {names[c]} | {fmt(asr, len(att))} | {fmt(mix, len(att))} | {fmt(blk, len(att))} | "
                         f"{fmt(cor, len(att))} | {fmt(cor_n, len(non))} | {fmt(blk_n, len(non))} |")
            rows.append([s, c, len(att), asr, blk, cor, len(non), cor_n, blk_n])
            chart[s][c] = 100 * asr / len(att) if att else 0
        # clean-corpus reference for the SAME queries that were attacked
        ids = [r["qid"] for r in att]
        c_cor = sum(clean[i].get(f"label_C_{a.labels}") == "correct" for i in ids if i in clean)
        c_false = sum(clean[i].get(f"label_C_{a.labels}") == "false" for i in ids if i in clean)
        lines.append(f"| Clean corpus (reference) | {fmt(c_false, len(ids))} | - | - | "
                     f"{fmt(c_cor, len(ids))} | - | - |")
        # detection view: did the agent flag the attacked context?
        flagged = sum(not r["trust_trustworthy"] for r in att)
        fp = sum(not r["trust_trustworthy"] for r in non)
        lines.append(f"\nDetection (Trust Index < 0.5): recall {fmt(flagged, len(att))}, "
                     f"false-alarm rate on untouched queries {fmt(fp, len(non))}\n")

    text = "\n".join(lines)
    print(text)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "asr_tables.md").write_text(text, encoding="utf-8")
    with open(out / "asr_summary.csv", "w", encoding="utf-8") as f:
        f.write("strategy,condition,n_attacked,asr_k,blocked_k,correct_k,n_untouched,correct_untouched_k,blocked_untouched_k\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        w = 0.8 / len(conds)
        for i, c in enumerate(conds):
            ax.bar([j + i * w for j in range(len(strategies))],
                   [chart[s][c] for s in strategies], w, label=names[c])
        ax.set_xticks([j + 0.4 - w / 2 for j in range(len(strategies))])
        ax.set_xticklabels(strategies)
        ax.set_ylabel("Attack success rate (%)")
        ax.set_title("ASR by attack and defense")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "asr_chart.png", dpi=200)
        print(f"\nSaved {out/'asr_tables.md'}, {out/'asr_summary.csv'}, {out/'asr_chart.png'}")
    except ImportError:
        print("matplotlib missing; skipped chart")


if __name__ == "__main__":
    main()
