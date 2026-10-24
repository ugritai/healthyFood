#!/usr/bin/env python3
"""
Rejection of absent foods on real LLM extractions (no annotators). Needs Neo4j.

Input: the distinct ingredient names extracted by Qwen3-8B (or the model given
with --model) from the 100 held-out RecipeNLG recipes
(data/extractions_held-out.jsonl).

For each name q, its content words are taken (lower case, letters only, stop
words removed; plural -s/-es/-ies also tried). Every entry of the matcher's
pool (processed entries excluded, as in the app) whose name contains any of
these words as a substring is removed, so the food is absent from the KG by
construction while the rest of the KG stays. The matcher is
then run on q against the remaining pool; the correct decision is to reject
(S < tau). Words of fewer than 3 letters are ignored.

Known limitation: synonyms written with a different word stay in the pool
(e.g. cilantro / Coriander); the accepted cases are listed for inspection.

Outputs
    results/negatives_items.csv    one row per name: words used, entries removed,
                                   top-1 after removal and its score, rejected
    results/negatives_results.csv  rejection rate at tau, per name and per mention
    results/negatives_sweep.csv    rejection rate over tau

Usage
    cd evaluation/linking
    python negatives_ablation.py          # Qwen3-8B, tau = 0.68
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

LINKING_DIR = Path(__file__).parent
sys.path.insert(0, str(LINKING_DIR.parent))
from evaluate import _connect_neo4j, _load_food_df  # noqa: E402

STOP = {"and", "or", "of", "with", "the", "a", "an", "to", "for", "in", "on"}


def word_forms(w: str) -> set[str]:
    forms = {w}
    if w.endswith("ies"):
        forms |= {w[:-3] + "y", w[:-3] + "i", w[:-1]}
    elif w.endswith("es"):
        forms |= {w[:-2], w[:-1]}
    elif w.endswith("s") and not w.endswith("ss"):
        forms.add(w[:-1])
    return forms


def name_forms(name: str) -> set[str]:
    words = [w for w in re.findall(r"[a-z]+", name.lower()) if w not in STOP]
    return {f for w in words for f in word_forms(w) if len(f) >= 3}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tau", type=float, default=0.68)
    ap.add_argument("--model", default="qwen3:8b-q8_0",
                    help="use only the extractions of this model; 'all' for the five models")
    ap.add_argument("--neo4j-uri", default="bolt://localhost:17687")
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-pass", default="TFGAmadeo")
    args = ap.parse_args()
    res = LINKING_DIR / "results"
    tag = "" if args.model == "qwen3:8b-q8_0" else f"_{args.model.split(':')[0]}"

    mentions = Counter()
    for line in (LINKING_DIR / "data" / "extractions_held-out.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if args.model != "all" and r["model"] != args.model:
            continue
        for n in r["ingredients"]:
            if n.strip():
                mentions[n.strip().lower()] += 1
    names = sorted(mentions)

    driver = _connect_neo4j(args.neo4j_uri, args.neo4j_user, args.neo4j_pass)
    df = _load_food_df(driver)
    driver.close()
    pool = df[~df["is_processed"]]["food_name"].tolist()
    pool_lower = [f.lower() for f in pool]

    from rapidfuzz.distance import JaroWinkler
    from rapidfuzz.process import cdist
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_pool = model.encode(pool, convert_to_numpy=True, normalize_embeddings=True)
    emb_q = model.encode(names, convert_to_numpy=True, normalize_embeddings=True)
    S = 0.7 * (emb_q @ emb_pool.T) + 0.3 * cdist(names, pool_lower, scorer=JaroWinkler.normalized_similarity, workers=-1)

    rows = []
    for i, q in enumerate(names):
        forms = name_forms(q)
        if not forms:
            continue
        removed = np.array([any(f in p for f in forms) for p in pool_lower])
        full_top = int(np.argmax(S[i]))
        masked = np.where(removed, -np.inf, S[i])
        top = int(np.argmax(masked))
        rows.append({"ingredient": q, "n_mentions": mentions[q], "words": ";".join(sorted(forms)),
                     "n_removed": int(removed.sum()),
                     "top1_full_kg": pool[full_top], "s_full_kg": round(float(S[i, full_top]), 4),
                     "top1_after_removal": pool[top], "s_after_removal": round(float(S[i, top]), 4),
                     "rejected": bool(S[i, top] < args.tau)})
    items = pd.DataFrame(rows).sort_values("s_after_removal", ascending=False)
    items.to_csv(res / f"negatives_items{tag}.csv", index=False)

    def rate(d: pd.DataFrame, tau: float, w: str | None) -> float:
        wt = d[w] if w else pd.Series(1, index=d.index)
        return 100 * wt[d["s_after_removal"] < tau].sum() / wt.sum()

    summary = []
    for subset, d in (("all", items), ("some entry removed (n_removed > 0)", items[items["n_removed"] > 0])):
        for unit, w in (("names", None), ("mentions", "n_mentions")):
            summary.append({"subset": subset, "unit": unit, "tau": args.tau,
                            "n": int(d[w].sum()) if w else len(d), "rejected_pct": round(rate(d, args.tau, w), 1)})
    summary = pd.DataFrame(summary)
    summary.to_csv(res / f"negatives_results{tag}.csv", index=False)
    sweep = pd.DataFrame([{"tau": t, "rejected_pct": round(rate(items, t, None), 1)}
                          for t in np.round(np.arange(0.50, 0.96, 0.02), 2)])
    sweep.to_csv(res / f"negatives_sweep{tag}.csv", index=False)

    print(f"{len(names)} distinct names, {len(items)} with content words of >= 3 letters; "
          f"some entry removed for {(items['n_removed'] > 0).sum()}")
    print(summary.to_string(index=False))
    s = items["s_after_removal"]
    print(f"\nS after removal: median {s.median():.3f}, IQR {s.quantile(.25):.3f}-{s.quantile(.75):.3f}")
    print(f"\nSweep (names):\n{sweep.to_string(index=False)}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 50)
    acc = items[~items["rejected"]]
    print(f"\nAccepted after removal (S >= {args.tau}), {len(acc)}:")
    print(acc[["ingredient", "words", "n_removed", "top1_after_removal", "s_after_removal"]].to_string(index=False))


if __name__ == "__main__":
    main()
