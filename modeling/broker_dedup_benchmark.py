#!/usr/bin/env python3
"""
Broker Research Near-Duplicate Dedup Benchmark
==============================================
Generates synthetic broker research notes (~300–500 tokens), injects near-duplicates,
and benchmarks four methods:
  1) SimHash (char-5-grams, 64 bits)          -> threshold on Hamming distance (lower = more similar)
  2) Exact Jaccard (char-5 sets)               -> threshold on similarity (higher = more similar)
  3) MinHash-estimated Jaccard (char-5, 128h)  -> threshold on similarity
  4) TF-IDF cosine (word tokens)               -> threshold on similarity

Outputs:
  - summary.csv            : best F1 thresholds & scores for each method
  - clusters_preview.csv   : a few predicted clusters from the top-scoring method
  - pairwise_sample.csv    : random sample of pairwise scores for inspection

Usage:
  python broker_dedup_benchmark.py --n-base 30 --variants 3 --seed 7 --outdir ./out

Dependencies:
  - Python 3.9+
  - numpy, pandas

Install:
  pip install numpy pandas
"""

import argparse
import hashlib
import itertools as it
import math
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# ------------------------- Text helpers -------------------------

def word_tokens(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())

def char_ngrams(text: str, n=5):
    s = re.sub(r"\s+", " ", text.lower()).strip()
    s = f"^{s}$"
    return [s[i:i+n] for i in range(max(0, len(s)-n+1))]

def md5_int(x: str) -> int:
    return int(hashlib.md5(x.encode("utf-8")).hexdigest(), 16)


# ------------------------- SimHash -------------------------

def simhash(features, weights=None, bits=64, hash_fn=md5_int) -> int:
    if weights is None:
        weights = Counter(features)
    else:
        weights = Counter({f: weights.get(f, 1.0) for f in features})
    acc = [0.0] * bits
    for f, w in weights.items():
        h = hash_fn(f)
        # fold to target width
        if h.bit_length() > bits:
            folded = 0
            while h:
                folded ^= (h & ((1 << bits) - 1))
                h >>= bits
            h = folded
        else:
            h &= (1 << bits) - 1
        for i in range(bits):
            acc[i] += w if (h >> i) & 1 else -w
    sig = 0
    for i, v in enumerate(acc):
        if v >= 0:
            sig |= (1 << i)
    return sig

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ------------------------- MinHash (for Jaccard) -------------------------

LARGE_PRIME = 2_147_483_647  # 2^31-1

def make_minhash_params(k=128, max_hash=LARGE_PRIME-1, rng=None):
    rng = rng or random
    params = []
    for _ in range(k):
        a = rng.randint(1, max_hash-1)
        b = rng.randint(0, max_hash-1)
        params.append((a, b))
    return params

def minhash_signature(int_shingles, params, max_hash=LARGE_PRIME-1):
    sig = []
    for a, b in params:
        m = min(((a*x + b) % LARGE_PRIME) for x in int_shingles) if int_shingles else max_hash
        sig.append(m)
    return tuple(sig)

def minhash_similarity(sigA, sigB):
    eq = sum(1 for a, b in zip(sigA, sigB) if a == b)
    return eq / len(sigA)


# ------------------------- TF-IDF Cosine -------------------------

def build_tfidf(corpus_tokens):
    N = len(corpus_tokens)
    df = Counter()
    for toks in corpus_tokens:
        df.update(set(toks))
    idf = {t: math.log(1 + N / (1 + df[t])) for t in df}
    vecs = []
    for toks in corpus_tokens:
        tf = Counter(toks)
        vec = {t: (tf[t] * idf.get(t, 0.0)) for t in tf}
        # L2 normalize
        norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
        for t in list(vec.keys()):
            vec[t] /= norm
        vecs.append(vec)
    return vecs

def cosine_sparse(vecA: dict, vecB: dict):
    # dot over the smaller dict
    if len(vecA) > len(vecB):
        vecA, vecB = vecB, vecA
    return sum(v * vecB.get(t, 0.0) for t, v in vecA.items())


# ------------------------- Synthetic broker research generator -------------------------

SECTORS = ["Technology", "Healthcare", "Financials", "Energy", "Consumer", "Industrials"]
EVENTS = [
    "Q2 earnings beat consensus", "guidance raised for FY", "new product launch",
    "regulatory approval", "major customer win", "cost optimization program",
    "CEO transition", "share buyback", "margin expansion", "pricing power sustained"
]
VERBS = ["reiterate", "maintain", "upgrade", "downgrade", "initiate coverage with"]
RATINGS = ["BUY", "HOLD", "SELL"]
RISKS = [
    "macro headwinds", "execution risk", "competitive intensity", "FX volatility",
    "supply chain constraints", "regulatory uncertainty", "commodity prices"
]
ADJ = ["solid", "robust", "muted", "mixed", "constructive", "cautious", "encouraging"]
SYNONYMS = {
    "company": ["firm", "business", "issuer", "enterprise"],
    "expects": ["anticipates", "foresees", "projects"],
    "growth": ["expansion", "increase", "uptick"],
    "margin": ["profitability", "spread"],
    "guide": ["outlook", "guidance", "view"],
    "strong": ["solid", "robust"],
    "weak": ["soft", "muted"],
    "we": ["our team", "we continue to", "we still"],
    "believe": ["think", "view", "see"],
}

DISCLAIMER = (
    "This note is intended for institutional investors only. "
    "It is not investment advice. Past performance is not indicative of future results. "
    "Please see important disclosures and analyst certifications at the end of this report."
)

def make_company_name(idx):
    prefixes = ["Asterion", "Borealis", "Cordia", "Deltabyte", "Equinox", "Ferrovia",
                "Granite", "Helios", "Icarus", "Juniper", "Krypton", "Lattice",
                "Monarch", "Nimbus", "Orion", "Pinnacle", "Quasar", "Radiant",
                "Solace", "Titan", "Umbra", "Vanguard", "Willow", "Xenon", "Yttria", "Zenith"]
    suffixes = ["Technologies", "Pharma", "Financial", "Energy", "Consumer", "Industries"]
    return f"{prefixes[idx % len(prefixes)]} {suffixes[idx % len(suffixes)]}"

def make_ticker(idx):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(alphabet[(idx*7 + i) % 26] for i in range(3))

def replace_synonyms(text, rng):
    def repl(m):
        w = m.group(0)
        choices = SYNONYMS.get(w, [w])
        return rng.choice(choices)
    if not SYNONYMS:
        return text
    pattern = r"\b(" + "|".join(map(re.escape, SYNONYMS.keys())) + r")\b"
    return re.sub(pattern, repl, text)

def add_typos(text, rng, prob=0.003):
    out = []
    for ch in text:
        if rng.random() < prob and ch.isalpha():
            if rng.random() < 0.5:
                continue   # drop
            else:
                out.append(ch)  # duplicate
        out.append(ch)
    return "".join(out)

def shuffle_sentences(paragraph, rng):
    sents = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    rng.shuffle(sents)
    return " ".join(sents)

def make_base_note(idx, rng):
    company = make_company_name(idx)
    ticker = make_ticker(idx)
    sector = rng.choice(SECTORS)
    event = rng.choice(EVENTS)
    verb = rng.choice(VERBS)
    rating = rng.choice(RATINGS)
    pt = round(rng.uniform(10, 500), 2)
    risks = ", ".join(rng.sample(RISKS, 3))
    adj1, adj2 = rng.sample(ADJ, 2)

    p1 = (f"We {verb} {rating} on {company} ({ticker}) after {event}. "
          f"Our {sector} team highlights {adj1} execution and {adj2} demand. "
          f"Price target {pt}.")
    p2 = (f"The company expects continued growth driven by product cycles and mix. "
          f"We model revenue acceleration and margin expansion into next year. "
          f"Management commentary indicates pricing power and disciplined opex.")
    p3 = (f"Valuation: shares trade at a discount to peers on EV/EBITDA and P/E. "
          f"We believe multiple expansion is possible as visibility improves.")
    p4 = (f"Key risks include {risks}. We stress test scenarios under different cost structures.")
    paragraphs = [p1, p2, p3, p4, DISCLAIMER]

    filler = (f"{company} operates within the {sector} sector. "
              f"Our channel checks suggest demand trends that are {adj1}. "
              f"This analysis synthesizes public filings, management calls, and third-party data. ")
    while len(word_tokens(' '.join(paragraphs))) < rng.randint(320, 420):
        paragraphs.insert(-1, filler)

    return {
        "cluster_id": idx,
        "company": company,
        "ticker": ticker,
        "sector": sector,
        "rating": rating,
        "pt": pt,
        "text": "\n\n".join(paragraphs),
    }

def perturb_note(base, rng):
    text = base["text"]
    if rng.random() < 0.7:
        text = replace_synonyms(text, rng)
    if rng.random() < 0.6:
        text = add_typos(text, rng, prob=0.0025)
    if rng.random() < 0.5:
        text = shuffle_sentences(text, rng)
    if rng.random() < 0.5:
        text = text.replace("Price target", "PT").replace("We model", "We forecast")
    if rng.random() < 0.4:
        text = re.sub(r"\bEV/EBITDA\b", "EV to EBITDA", text)
    # Minor rating/PT tweaks to simulate confusing near-dupes
    rating = base["rating"]
    if rng.random() < 0.25:
        rating = rng.choice(RATINGS)
        text = re.sub(r"\b(BUY|HOLD|SELL)\b", rating, text, count=1)
    pt = base["pt"]
    if rng.random() < 0.25:
        delta = rng.uniform(-5, 5)
        pt2 = round(max(1.0, pt + delta), 2)
        text = re.sub(r"(Price target|PT) [0-9]+(\.[0-9]+)?", f"PT {pt2}", text, count=1)
    tokens = word_tokens(text)
    if len(tokens) < 300:
        text += "\n\n" + DISCLAIMER
    return {
        "cluster_id": base["cluster_id"],
        "company": base["company"],
        "ticker": base["ticker"],
        "sector": base["sector"],
        "rating": rating,
        "pt": pt,
        "text": text,
    }


# ------------------------- Benchmark core -------------------------

def best_f1_similarity(scores: np.ndarray, y_true: np.ndarray):
    order = np.argsort(-scores)  # desc
    y = y_true[order]
    tp_cum = np.cumsum(y)
    fp_cum = np.cumsum(1 - y)
    total_pos = int(tp_cum[-1]) if len(tp_cum) else 0
    prec = np.divide(tp_cum, tp_cum + fp_cum, out=np.zeros_like(tp_cum, dtype=float), where=(tp_cum+fp_cum)!=0)
    rec = np.divide(tp_cum, total_pos, out=np.zeros_like(tp_cum, dtype=float), where=total_pos!=0)
    f1 = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(prec), where=(prec+rec)!=0)
    idx = int(np.nanargmax(f1)) if len(f1) else 0
    thr = float(scores[order][idx]) if len(scores) else 0.0
    return float(thr), float(prec[idx]), float(rec[idx]), float(f1[idx])

def best_f1_distance(scores: np.ndarray, y_true: np.ndarray):
    order = np.argsort(scores)  # asc
    y = y_true[order]
    tp_cum = np.cumsum(y)
    fp_cum = np.cumsum(1 - y)
    total_pos = int(tp_cum[-1]) if len(tp_cum) else 0
    prec = np.divide(tp_cum, tp_cum + fp_cum, out=np.zeros_like(tp_cum, dtype=float), where=(tp_cum+fp_cum)!=0)
    rec = np.divide(tp_cum, total_pos, out=np.zeros_like(tp_cum, dtype=float), where=total_pos!=0)
    f1 = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(prec), where=(prec+rec)!=0)
    idx = int(np.nanargmax(f1)) if len(f1) else 0
    thr = float(scores[order][idx]) if len(scores) else 0.0
    return float(thr), float(prec[idx]), float(rec[idx]), float(f1[idx])


def clusters_from_edges(N, scores, thr, mode="similarity"):
    parent = list(range(N))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    # scores is aligned to the list of (i,j) pairs in lexicographic order
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            s = scores[idx]
            ok = (s >= thr) if mode == "similarity" else (s <= thr)
            if ok: union(i, j)
            idx += 1
    clusters = defaultdict(list)
    for i in range(N):
        clusters[find(i)].append(i)
    return [sorted(v) for v in clusters.values() if len(v) > 1]


def run_benchmark(args):
    rng = random.Random(args.seed)

    # ---- Build corpus ----
    corpus = []
    for i in range(args.n_base):
        base = make_base_note(i, rng)
        variants = [base] + [perturb_note(base, rng) for _ in range(args.variants - 1)]
        corpus.extend(variants)

    docs = [d["text"] for d in corpus]
    cluster_ids = np.array([d["cluster_id"] for d in corpus], dtype=int)
    N = len(docs)
    print(f"Built synthetic corpus with {N} documents "
          f"({args.n_base} companies × {args.variants} variants).")

    # ---- Features ----
    # SimHash
    sim_sigs = []
    for txt in docs:
        feats = char_ngrams(txt, n=5)
        sim_sigs.append(simhash(feats, weights=Counter(feats), bits=64))

    # Char-5 sets + MinHash integers
    shingle_sets = []
    for txt in docs:
        c5 = set(char_ngrams(txt, n=5))
        ints = [md5_int(s) % LARGE_PRIME for s in c5]
        shingle_sets.append((c5, ints))

    # MinHash signatures
    mh_params = make_minhash_params(k=128, rng=rng)
    mh_sigs = [minhash_signature(ints, mh_params) for (_, ints) in shingle_sets]

    # TF-IDF vectors
    token_lists = [word_tokens(t) for t in docs]
    tfidf_vecs = build_tfidf(token_lists)

    # ---- Pairwise metrics ----
    n_pairs = N*(N-1)//2
    simhash_ham = np.empty(n_pairs, dtype=np.int32)
    simhash_sim = np.empty(n_pairs, dtype=np.float32)
    jaccard = np.empty(n_pairs, dtype=np.float32)
    minhash_est = np.empty(n_pairs, dtype=np.float32)
    cosine = np.empty(n_pairs, dtype=np.float32)
    y_true = np.empty(n_pairs, dtype=np.int8)

    k = 0
    for i in range(N):
        for j in range(i+1, N):
            same = 1 if cluster_ids[i] == cluster_ids[j] else 0
            ham = hamming(sim_sigs[i], sim_sigs[j])
            sim_simhash = 1.0 - ham / 64.0
            A = shingle_sets[i][0]; B = shingle_sets[j][0]
            jac = len(A & B) / max(1, len(A | B))
            mh_sim = minhash_similarity(mh_sigs[i], mh_sigs[j])
            cos = cosine_sparse(tfidf_vecs[i], tfidf_vecs[j])

            y_true[k] = same
            simhash_ham[k] = ham
            simhash_sim[k] = sim_simhash
            jaccard[k] = jac
            minhash_est[k] = mh_sim
            cosine[k] = cos
            k += 1

    # ---- Threshold sweeps ----
    thr_ham, p_ham, r_ham, f1_ham = best_f1_distance(simhash_ham, y_true)
    thr_jac, p_jac, r_jac, f1_jac = best_f1_similarity(jaccard, y_true)
    thr_mh,  p_mh,  r_mh,  f1_mh  = best_f1_similarity(minhash_est, y_true)
    thr_cos, p_cos, r_cos, f1_cos = best_f1_similarity(cosine, y_true)

    summary = pd.DataFrame([
        {"method": "TF-IDF cosine (words)", "threshold_note": "Cosine ≥ t", "threshold": thr_cos, "precision": p_cos, "recall": r_cos, "f1": f1_cos},
        {"method": "SimHash (char-5)", "threshold_note": "Hamming ≤ t", "threshold": thr_ham, "precision": p_ham, "recall": r_ham, "f1": f1_ham},
        {"method": "Jaccard exact (char-5 set)", "threshold_note": "Jaccard ≥ t", "threshold": thr_jac, "precision": p_jac, "recall": r_jac, "f1": f1_jac},
        {"method": "MinHash est. (char-5)", "threshold_note": "Est. Jaccard ≥ t", "threshold": thr_mh, "precision": p_mh, "recall": r_mh, "f1": f1_mh},
    ]).sort_values("f1", ascending=False).reset_index(drop=True)

    os.makedirs(args.outdir, exist_ok=True)
    summary_path = os.path.join(args.outdir, "summary.csv")
    summary.to_csv(summary_path, index=False)

    # ---- Clusters from top method ----
    top = summary.iloc[0]["method"]
    if "SimHash" in top:
        clusters = clusters_from_edges(N, simhash_ham, thr_ham, mode="distance")
        score_note = "Hamming distance"
    elif "TF-IDF" in top:
        clusters = clusters_from_edges(N, cosine, thr_cos, mode="similarity")
        score_note = "Cosine similarity"
    elif "MinHash" in top:
        clusters = clusters_from_edges(N, minhash_est, thr_mh, mode="similarity")
        score_note = "MinHash-estimated Jaccard"
    else:
        clusters = clusters_from_edges(N, jaccard, thr_jac, mode="similarity")
        score_note = "Exact Jaccard (char-5)"

    preview_rows = []
    for cid, cl in enumerate(clusters[:6], start=1):
        for i in cl:
            preview_rows.append({
                "cluster_id": cid,
                "doc_id": i,
                "tokens": len(word_tokens(docs[i])),
                "ticker": corpus[i]["ticker"],
                "rating": corpus[i]["rating"],
                "truth_cluster": corpus[i]["cluster_id"],
                "snippet": re.sub(r"\s+", " ", docs[i])[:140] + "...",
                "full": re.sub(r"\s+", " ", docs[i]),
            })
    preview_df = pd.DataFrame(preview_rows)
    preview_path = os.path.join(args.outdir, "clusters_preview.csv")
    preview_df.to_csv(preview_path, index=False)

    # ---- Pairwise sample ----
    # Save a random sample of pairwise scores for inspection
    df_pairs = pd.DataFrame({
        "same_cluster": y_true.astype(int),
        "simhash_hamming": simhash_ham.astype(int),
        "simhash_sim": simhash_sim.astype(float),
        "jaccard": jaccard.astype(float),
        "minhash_est": minhash_est.astype(float),
        "cosine": cosine.astype(float),
    })
    # Reconstruct i,j for readability (optional)
    I, J = [], []
    for i in range(N):
        for j in range(i+1, N):
            I.append(i); J.append(j)
    df_pairs.insert(0, "i", I)
    df_pairs.insert(1, "j", J)

    sample = df_pairs.sample(min(args.pairwise_sample, len(df_pairs)), random_state=args.seed)
    pairs_path = os.path.join(args.outdir, "pairwise_sample.csv")
    sample.to_csv(pairs_path, index=False)

    # ---- Print results ----
    print("\n=== Dedup Benchmark Summary (best F1 thresholds) ===")
    with pd.option_context('display.max_colwidth', None):
        print(summary.to_string(index=False))
    print(f"\nSaved results to:\n  {summary_path}\n  {preview_path}\n  {pairs_path}")
    print(f"\nTop method: {top} (score: {score_note})")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-base", type=int, default=30, help="Number of base companies.")
    ap.add_argument("--variants", type=int, default=3, help="Variants per company (including base).")
    ap.add_argument("--seed", type=int, default=7, help="Random seed.")
    ap.add_argument("--outdir", type=str, default="./out", help="Directory for outputs.")
    ap.add_argument("--pairwise-sample", type=int, default=400, help="Rows to write to pairwise_sample.csv")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
