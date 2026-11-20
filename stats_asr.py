"""
stats_asr.py -- descriptive and inferential statistics for the ASR experiment.

Usage:  python stats_asr.py --in data/asr/asr_results.jsonl [--labels h|j] [--boot 10000]

Reports, per attack strategy:
  1. Label distribution per condition (correct / false / mixed / neither / blocked)
  2. Paired comparison of ASR: A (no defense) vs B, G, R
       - ASR difference with paired-bootstrap 95% CI
       - exact McNemar test (two-sided) on discordant pairs, Holm-corrected
  3. Trust Index descriptives (mean, SD, median) for attacked vs untouched queries,
     plus ROC-AUC of the Trust Index as an attacked-vs-untouched detector
  4. Cost of defense on untouched queries (answers lost to blocking / filtering)
  5. Agreement between heuristic and LLM-judge labels (Cohen's kappa), if judge labels exist
"""
import argparse
import json
import random
import statistics as st
from collections import Counter
from math import comb

LABELS = ["correct", "false", "mixed", "neither", "blocked"]


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def paired_bootstrap_diff(x, y, boot=10000, seed=0):
    """x, y: paired 0/1 lists. Returns (mean(y)-mean(x), lo, hi) with percentile 95% CI."""
    n = len(x)
    if n == 0:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    diffs = []
    for _ in range(boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(y[i] - x[i] for i in idx) / n)
    diffs.sort()
    return (sum(y) / n - sum(x) / n, diffs[int(0.025 * boot)], diffs[int(0.975 * boot) - 1])


def auc_lower_is_positive(pos_scores, neg_scores):
    """AUC where a LOWER score means 'positive' (attacked). Mann-Whitney with ties = 0.5."""
    if not pos_scores or not neg_scores:
        return float("nan")
    wins = 0.0
    for p in pos_scores:
        for q in neg_scores:
            wins += 1.0 if p < q else 0.5 if p == q else 0.0
    return wins / (len(pos_scores) * len(neg_scores))


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def mean_sd(v):
    if not v:
        return "n/a"
    sd = st.stdev(v) if len(v) > 1 else 0.0
    return f"{st.mean(v):.3f} +/- {sd:.3f} (median {st.median(v):.3f}, n={len(v)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/asr/asr_results.jsonl")
    ap.add_argument("--labels", choices=["h", "j"], default="h")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    strategies = sorted({r["strategy"] for r in recs if r["strategy"] != "clean"})
    conds = ["A", "B", "G"] + [c for c in ("R", "R2", "R3") if any(f"label_{c}_h" in r for r in recs)]
    lab = lambda r, c: r.get(f"label_{c}_{a.labels}")
    out = [f"Labels: {'heuristic' if a.labels == 'h' else 'LLM judge'}\n"]

    for s in strategies:
        rs = [r for r in recs if r["strategy"] == s]
        att = [r for r in rs if r["attacked"]]
        non = [r for r in rs if not r["attacked"]]
        out.append(f"## Attack: {s} (attacked n={len(att)}, untouched n={len(non)})\n")

        out.append("### 1. Label distribution on attacked queries")
        out.append("| Condition | " + " | ".join(LABELS) + " |")
        out.append("|---|" + "---|" * len(LABELS))
        for c in conds:
            cnt = Counter(lab(r, c) for r in att)
            out.append(f"| {c} | " + " | ".join(str(cnt.get(l, 0)) for l in LABELS) + " |")

        out.append("\n### 2. Paired ASR comparison vs A (no defense)")
        out.append("| Defense | ASR A | ASR def | Diff [95% CI] | b (A-only success) | c (def-only success) | McNemar p | Holm p |")
        out.append("|---|---|---|---|---|---|---|---|")
        rows, pvals = [], []
        for c in conds[1:]:
            x = [int(lab(r, "A") == "false") for r in att]
            y = [int(lab(r, c) == "false") for r in att]
            b = sum(1 for i, j in zip(x, y) if i == 1 and j == 0)
            cc = sum(1 for i, j in zip(x, y) if i == 0 and j == 1)
            d, lo, hi = paired_bootstrap_diff(x, y, a.boot)
            p = mcnemar_exact(b, cc)
            rows.append((c, sum(x), sum(y), len(att), d, lo, hi, b, cc, p))
            pvals.append(p)
        adj = holm(pvals) if pvals else []
        for (c, kx, ky, n, d, lo, hi, b, cc, p), pa in zip(rows, adj):
            out.append(f"| {c} | {kx}/{n} | {ky}/{n} | {100*d:+.1f} pp [{100*lo:+.1f}, {100*hi:+.1f}] "
                       f"| {b} | {cc} | {p:.4f} | {pa:.4f} |")

        out.append("\n### 3. Trust Index descriptives (condition A response, K retrieved docs)")
        ta = [r["trust_score"] for r in att]
        tn = [r["trust_score"] for r in non]
        out.append(f"- attacked : {mean_sd(ta)}")
        out.append(f"- untouched: {mean_sd(tn)}")
        out.append(f"- separation (untouched - attacked mean): "
                   f"{(st.mean(tn) - st.mean(ta)):.3f}" if ta and tn else "- separation: n/a")
        out.append(f"- ROC-AUC (low trust => attacked): {auc_lower_is_positive(ta, tn):.3f}")
        mp = [max(r["doc_poison_probs"]) for r in att if r["doc_poison_probs"]]
        mn = [max(r["doc_poison_probs"]) for r in non if r["doc_poison_probs"]]
        out.append(f"- max per-doc poison prob, attacked : {mean_sd(mp)}")
        out.append(f"- max per-doc poison prob, untouched: {mean_sd(mn)}")

        out.append("\n### 4. Cost of defense on untouched queries")
        out.append("| Condition | Blocked | Correct | Correct-rate change vs A (pp) |")
        out.append("|---|---|---|---|")
        base = sum(lab(r, "A") == "correct" for r in non)
        for c in conds:
            blk = sum(lab(r, c) == "blocked" for r in non)
            cor = sum(lab(r, c) == "correct" for r in non)
            delta = 100 * (cor - base) / len(non) if non else 0.0
            out.append(f"| {c} | {blk}/{len(non)} | {cor}/{len(non)} | {delta:+.1f} |")
        out.append("")

    if any(r.get("label_A_j") for r in recs):
        hs, js = [], []
        for r in recs:
            for c in ("A", "B", "G", "R", "R2", "R3", "C"):
                h, j = r.get(f"label_{c}_h"), r.get(f"label_{c}_j")
                if h and j and h != "blocked":
                    hs.append(h)
                    js.append(j)
        agree = sum(x == y for x, y in zip(hs, js)) / max(1, len(hs))
        out.append(f"## 5. Heuristic vs LLM-judge agreement\n"
                   f"n={len(hs)}, raw agreement={agree:.2%}, Cohen's kappa={cohen_kappa(hs, js):.3f}")
        out.append("(Heuristic uses 'mixed'; the judge never does, so some disagreement is structural.)")

    text = "\n".join(out)
    print(text)
    open(a.inp.rsplit(".", 1)[0] + f"_stats_{a.labels}.md", "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
