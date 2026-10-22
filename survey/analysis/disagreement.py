"""
Learning from disagreement: aggregation methods that model annotator
disagreement instead of discarding it, plus per-relation soft labels.

Methods (all reported, whatever direction they go):
- crowdtruth   CrowdTruth 2.0 (Dumitrache et al., 2018): worker and unit
               quality scores estimated jointly; Not sure is kept as a
               legitimate annotation. A relation is accepted when its
               quality-weighted Agree score exceeds its Disagree score.
- mace         MACE (Hovy et al., 2013), binary, on decisive answers: each
               annotator either reports the truth (with probability
               theta_j) or answers from a personal "spam" distribution.
- soft         Soft accuracy: mean over relations of the share of Agree
               among decisive answers (no hard verdict).

Also exports soft labels (vote distribution, entropy, CrowdTruth unit
scores) per relation, which can be attached to graph edges as a consensus
score.

Usage:
    python disagreement.py [path/to/export.xlsx]
"""

import numpy as np
import pandas as pd

from aggregation import majority_vote
from common import (
    CORRECTNESS_ORDER,
    TYPE_LABELS,
    TYPE_ORDER,
    SurveyData,
    load,
    normalized_entropy,
    pct,
    rating_matrix,
    save_table,
    section,
    xlsx_from_argv,
)
from decision_confidence import wilson


# --------------------------------------------------------------------------
# CrowdTruth 2.0
# --------------------------------------------------------------------------
def crowdtruth(items: pd.DataFrame, n_iter: int = 50) -> tuple[pd.DataFrame, pd.Series]:
    """Returns unit-annotation scores per relation (columns = answers) and
    worker quality scores. Each judgment is a one-hot vector over the three
    answers; annotation quality is kept uniform."""
    cats = CORRECTNESS_ORDER
    units = sorted(items.item_id.unique())
    workers = sorted(items.respondent_id.unique())
    ui = {u: i for i, u in enumerate(units)}
    wi = {w: i for i, w in enumerate(workers)}
    v = np.zeros((len(units), len(workers), len(cats)))
    for r in items.itertuples():
        v[ui[r.item_id], wi[r.respondent_id], cats.index(r.label)] = 1
    rated = v.sum(axis=2) > 0  # (units, workers)

    wqs = np.ones(len(workers))
    uqs = np.ones(len(units))
    for _ in range(n_iter):
        # unit quality: weighted pairwise agreement between the workers of each unit
        new_uqs = np.zeros(len(units))
        for u in range(len(units)):
            ws = np.where(rated[u])[0]
            num = den = 0.0
            for a in range(len(ws)):
                for b in range(a + 1, len(ws)):
                    w = wqs[ws[a]] * wqs[ws[b]]
                    num += w * float(v[u, ws[a]] @ v[u, ws[b]])
                    den += w
            new_uqs[u] = num / den if den else 0.0
        # worker quality: agreement with the unit (minus themselves) x agreement with other workers
        wua = np.zeros(len(workers))
        wwa = np.zeros(len(workers))
        for w in range(len(workers)):
            us = np.where(rated[:, w])[0]
            ua_num = ua_den = wa_num = wa_den = 0.0
            for u in us:
                others = np.where(rated[u])[0]
                others = others[others != w]
                if not len(others):
                    continue
                unit_vec = (v[u, others] * wqs[others, None]).sum(axis=0)
                norm = np.linalg.norm(unit_vec)
                cos = float(v[u, w] @ unit_vec) / norm if norm else 0.0
                ua_num += cos * new_uqs[u]
                ua_den += new_uqs[u]
                pair = (v[u, others] @ v[u, w]) * wqs[others]
                wa_num += pair.sum() * new_uqs[u]
                wa_den += wqs[others].sum() * new_uqs[u]
            wua[w] = ua_num / ua_den if ua_den else 0.0
            wwa[w] = wa_num / wa_den if wa_den else 0.0
        new_wqs = wua * wwa
        converged = np.allclose(new_wqs, wqs, atol=1e-6) and np.allclose(new_uqs, uqs, atol=1e-6)
        wqs, uqs = new_wqs, new_uqs
        if converged:
            break

    uas = (v * wqs[None, :, None]).sum(axis=1)
    uas = uas / np.where(uas.sum(axis=1, keepdims=True) > 0, uas.sum(axis=1, keepdims=True), 1)
    scores = pd.DataFrame(uas, index=units, columns=[f"ct_{c}" for c in cats])
    scores["ct_unit_quality"] = uqs
    return scores, pd.Series(wqs, index=workers, name="ct_worker_quality")


# --------------------------------------------------------------------------
# MACE (binary, decisive answers)
# --------------------------------------------------------------------------
def mace(items: pd.DataFrame, n_iter: int = 200, n_restarts: int = 10, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    dec = items[items.label != "not_sure"]
    units = sorted(dec.item_id.unique())
    workers = sorted(dec.respondent_id.unique())
    u_idx = dec.item_id.map({u: i for i, u in enumerate(units)}).to_numpy()
    w_idx = dec.respondent_id.map({w: i for i, w in enumerate(workers)}).to_numpy()
    lab = (dec.label == "agree").astype(int).to_numpy()  # 1 = agree
    rng = np.random.default_rng(seed)

    best = None
    for _ in range(n_restarts):
        theta = rng.uniform(0.5, 0.9, len(workers))
        xi = rng.dirichlet([10, 10], len(workers))  # spam distribution over (disagree, agree)
        for _ in range(n_iter):
            # E-step: posterior over the true label of each unit
            log_post = np.zeros((len(units), 2))
            for t in (0, 1):
                p = theta[w_idx] * (lab == t) + (1 - theta[w_idx]) * xi[w_idx, lab]
                np.add.at(log_post[:, t], u_idx, np.log(p))
            post = np.exp(log_post - log_post.max(axis=1, keepdims=True))
            post /= post.sum(axis=1, keepdims=True)
            # M-step: expected counts of "truthful" vs "spamming" per annotator
            truth_num = np.zeros(len(workers))
            spam_cnt = np.full((len(workers), 2), 0.5)
            total = np.zeros(len(workers))
            for t in (0, 1):
                pt = post[u_idx, t]
                p_true = theta[w_idx] * (lab == t)
                p_spam = (1 - theta[w_idx]) * xi[w_idx, lab]
                r_true = pt * p_true / (p_true + p_spam)
                np.add.at(truth_num, w_idx, r_true)
                np.add.at(spam_cnt, (w_idx, lab), pt - r_true)
                np.add.at(total, w_idx, pt)
            theta = np.clip((truth_num + 0.5) / (total + 1.0), 1e-3, 1 - 1e-3)
            xi = spam_cnt / spam_cnt.sum(axis=1, keepdims=True)
        marginal = np.log(np.exp(log_post - log_post.max(axis=1, keepdims=True)).sum(axis=1)) + log_post.max(axis=1)
        score = marginal.sum()
        if best is None or score > best[0]:
            best = (score, post.copy(), theta.copy())

    _, post, theta = best
    labels = pd.Series(np.where(post[:, 1] > 0.5, "accept", "reject"), index=units)
    return labels, pd.Series(theta, index=workers, name="mace_competence")


# --------------------------------------------------------------------------
def soft_labels(items: pd.DataFrame) -> pd.DataFrame:
    counts = rating_matrix(items, "item_id", CORRECTNESS_ORDER)
    shares = counts.div(counts.sum(axis=1), axis=0)
    out = items.groupby("item_id")[["type", "food", "target"]].first()
    out[["p_agree", "p_not_sure", "p_disagree"]] = shares[CORRECTNESS_ORDER].round(3).to_numpy()
    decisive = counts.agree + counts.disagree
    out["p_agree_decisive"] = (counts.agree / decisive.where(decisive > 0)).round(3)
    out["entropy"] = normalized_entropy(counts).round(3)
    out["majority_vote"] = majority_vote(counts)
    return out


def compare(items: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    soft = soft_labels(items)
    ct_scores, _ = crowdtruth(items)
    soft = soft.join(ct_scores.round(3))
    soft["crowdtruth"] = np.select(
        [soft.ct_agree > soft.ct_disagree, soft.ct_disagree > soft.ct_agree], ["accept", "reject"], "tie"
    )
    mace_labels, _ = mace(items)
    soft["mace"] = mace_labels

    rows = []
    for name, col in [("Majority vote (reference)", "majority_vote"), ("CrowdTruth 2.0", "crowdtruth"), ("MACE", "mace")]:
        for t in [*TYPE_ORDER, "all"]:
            s = soft[col] if t == "all" else soft.loc[soft.type == t, col]
            acc, rej = int((s == "accept").sum()), int((s == "reject").sum())
            lo, hi = wilson(acc, acc + rej)
            rows.append({"method": name, "type": t, "accepted": acc, "rejected": rej,
                         "undecided": len(s) - acc - rej, "accuracy": pct(acc / (acc + rej)), "ci": f"{lo}–{hi}"})
    for t in [*TYPE_ORDER, "all"]:
        s = soft.p_agree_decisive if t == "all" else soft.loc[soft.type == t, "p_agree_decisive"]
        rows.append({"method": "Soft accuracy (mean share of Agree)", "type": t, "accepted": np.nan,
                     "rejected": np.nan, "undecided": np.nan, "accuracy": pct(s.mean()), "ci": ""})
    return pd.DataFrame(rows), soft


def run(data: SurveyData) -> str:
    section("LEARNING FROM DISAGREEMENT")
    results, soft = compare(data.items)
    save_table(results, "disagreement_methods")
    save_table(soft.reset_index(), "soft_labels")
    print(results.to_string(index=False))
    overall = results[results.type == "all"][["method", "accepted", "rejected", "undecided", "accuracy", "ci"]]
    flips = soft[(soft.crowdtruth != soft.majority_vote) | (soft.mace != soft.majority_vote)][
        ["type", "food", "target", "p_agree", "p_not_sure", "p_disagree", "majority_vote", "crowdtruth", "mace"]
    ]
    return f"""## 8. Learning from disagreement

Methods that model annotator disagreement instead of discarding it. All are reported.

{overall.to_markdown(index=False)}

By relation type:

{results.pivot(index='method', columns='type', values='accuracy')[[*TYPE_ORDER, 'all']].rename(columns=TYPE_LABELS).to_markdown()}

### Relations whose verdict changes with the method

{flips.to_markdown() if len(flips) else 'None.'}

Soft labels per relation (vote shares, entropy, CrowdTruth unit scores): `results/soft_labels.csv`.
"""


if __name__ == "__main__":
    print(run(load(xlsx_from_argv())))
