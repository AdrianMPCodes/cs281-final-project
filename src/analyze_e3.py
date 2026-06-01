"""E3: per-model flip rates + cross-model agreement.

Run after E1 completes. For each decisions_*.jsonl in --inputs:
  1. run find_flips logic in-process -> pairs file
  2. compute per-model flip rate stratified by swap type
  3. compute pairwise Jaccard overlap of the flipped-pair sets across models
  4. emit a per-pair table flagging which models flipped on each
"""
import argparse, json, sys
from collections import Counter, defaultdict
from itertools import combinations

AGES    = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
GENDERS = ["female", "male", "non-binary"]
RACES   = ["white", "Black", "Asian", "Hispanic", "Native American"]


def build_pairs(rows):
    idx = {(r["qid"], r["age"], r["gender"], r["race"]): r for r in rows}
    qids = sorted({r["qid"] for r in rows})
    out = []

    for qid in qids:
        for g in GENDERS:
            for r_ in RACES:
                for a1, a2 in combinations(AGES, 2):
                    A = idx[(qid, a1, g, r_)]; B = idx[(qid, a2, g, r_)]
                    out.append(("age", qid, g, r_, a1, a2,
                                A["decision"], B["decision"]))
        for r_ in RACES:
            for a in AGES:
                for g1, g2 in combinations(GENDERS, 2):
                    A = idx[(qid, a, g1, r_)]; B = idx[(qid, a, g2, r_)]
                    out.append(("gender", qid, r_, a, g1, g2,
                                A["decision"], B["decision"]))
        for g in GENDERS:
            for a in AGES:
                for r1, r2 in combinations(RACES, 2):
                    A = idx[(qid, a, g, r1)]; B = idx[(qid, a, g, r2)]
                    out.append(("race", qid, g, a, r1, r2,
                                A["decision"], B["decision"]))
    return out


def is_flip(pair):
    """pair = (swap, qid, ..., a_dec, b_dec)"""
    a, b = pair[-2], pair[-1]
    return (a is not None) and (b is not None) and (a != b)


def pair_id(pair):
    """Stable key for cross-model comparison: (swap, qid, fixed_keys, a_val, b_val)."""
    swap = pair[0]
    qid = pair[1]
    # The non-swap-key positions encode the held-fixed axes.
    fixed = pair[2:-4]
    a_val, b_val = pair[-4], pair[-3]
    # Canonicalize order so (A=20,B=30) and (A=30,B=20) collide.
    if str(a_val) > str(b_val):
        a_val, b_val = b_val, a_val
    return (swap, qid, *fixed, a_val, b_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, nargs="+",
                    help="decisions_*.jsonl files, one per model")
    ap.add_argument("--out-pairs-dir", default="results",
                    help="where to write per-model pairs JSONL")
    ap.add_argument("--out-summary", default="results/e3_summary.json")
    args = ap.parse_args()

    per_model_flips = {}      # model_label -> set of pair_id
    per_model_summary = {}    # model_label -> dict
    pair_to_models = defaultdict(list)  # pair_id -> [model_label that flipped]

    for path in args.inputs:
        rows = []
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        # Dedup on the (qid,age,gender,race) key, keeping the LAST seen row.
        # Append-mode resume can introduce duplicates if a prior partial row
        # later succeeded -- the successful one is what we want.
        ix = {(r["qid"], r["age"], r["gender"], r["race"]): r for r in rows}
        rows = list(ix.values())
        # Skip rows whose decision didn't parse.
        rows_parsed = [r for r in rows if r.get("decision") in ("APPROVE","DENY")]

        # Need a full grid to enumerate pairs reliably.
        expected = 11 * 9 * 3 * 5
        if len(rows_parsed) != expected:
            print(f"WARN {path}: {len(rows_parsed)}/{expected} parsed rows -- "
                  f"pair enumeration will skip incomplete cells.", file=sys.stderr)
            # Use a defensive index; pair-enum will KeyError if any cell missing.
            # Build pairs only for cells with parseable decisions.
            present = {(r["qid"], r["age"], r["gender"], r["race"]) for r in rows_parsed}
            idx = {k: ix[k] for k in present}
            # Fall back: re-enumerate with strict guard.
            pairs = []
            qids = sorted({r["qid"] for r in rows_parsed})
            for qid in qids:
                for g in GENDERS:
                    for r_ in RACES:
                        for a1, a2 in combinations(AGES, 2):
                            k1 = (qid, a1, g, r_); k2 = (qid, a2, g, r_)
                            if k1 not in idx or k2 not in idx: continue
                            pairs.append(("age", qid, g, r_, a1, a2,
                                          idx[k1]["decision"], idx[k2]["decision"]))
                for r_ in RACES:
                    for a in AGES:
                        for g1, g2 in combinations(GENDERS, 2):
                            k1 = (qid, a, g1, r_); k2 = (qid, a, g2, r_)
                            if k1 not in idx or k2 not in idx: continue
                            pairs.append(("gender", qid, r_, a, g1, g2,
                                          idx[k1]["decision"], idx[k2]["decision"]))
                for g in GENDERS:
                    for a in AGES:
                        for r1, r2 in combinations(RACES, 2):
                            k1 = (qid, a, g, r1); k2 = (qid, a, g, r2)
                            if k1 not in idx or k2 not in idx: continue
                            pairs.append(("race", qid, g, a, r1, r2,
                                          idx[k1]["decision"], idx[k2]["decision"]))
        else:
            pairs = build_pairs(rows_parsed)

        # The first decisions row's `model` field carries the slug; fall back to filename.
        model_label = (rows_parsed[0].get("model") if rows_parsed else None) or path

        # Summary
        by_swap_total = Counter()
        by_swap_flips = Counter()
        flipped_ids = set()
        for p in pairs:
            sw = p[0]
            by_swap_total[sw] += 1
            if is_flip(p):
                by_swap_flips[sw] += 1
                pid = pair_id(p)
                flipped_ids.add(pid)
                pair_to_models[pid].append(model_label)

        per_model_flips[model_label] = flipped_ids
        per_model_summary[model_label] = {
            "source": path,
            "n_parsed": len(rows_parsed),
            "n_pairs": len(pairs),
            "n_flips_total": sum(by_swap_flips.values()),
            "by_swap": {
                sw: {"flips": by_swap_flips[sw], "total": by_swap_total[sw],
                     "rate": round(by_swap_flips[sw] / by_swap_total[sw], 4)
                            if by_swap_total[sw] else None}
                for sw in ("age", "gender", "race")
            },
            "overall_rate": round(sum(by_swap_flips.values()) / sum(by_swap_total.values()), 4)
                            if sum(by_swap_total.values()) else None,
        }

    # Pairwise Jaccard on flipped-pair sets.
    labels = sorted(per_model_flips)
    jaccard = {}
    for a, b in combinations(labels, 2):
        sa, sb = per_model_flips[a], per_model_flips[b]
        inter = len(sa & sb)
        union = len(sa | sb)
        jaccard[f"{a} vs {b}"] = {
            "intersection": inter,
            "union": union,
            "jaccard": round(inter / union, 4) if union else None,
        }

    # Shared-flips multiset: how many models flipped on each pair?
    multi = Counter(len(v) for v in pair_to_models.values())

    summary = {
        "per_model": per_model_summary,
        "pairwise_jaccard": jaccard,
        "shared_flips": {f"flipped_on_{k}_models": v for k, v in sorted(multi.items())},
        "n_distinct_flipped_pairs_any_model": len(pair_to_models),
    }
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    # Human-readable printout
    print(f"=== E3 summary (saved to {args.out_summary}) ===\n")
    print("Per-model flip rates:")
    print(f"  {'model':45s} {'overall':>8s}  {'age':>10s}  {'gender':>10s}  {'race':>10s}")
    for lbl in labels:
        s = per_model_summary[lbl]
        print(f"  {lbl:45s} {s['overall_rate']*100:7.2f}%  "
              f"{s['by_swap']['age']['rate']*100:9.2f}%  "
              f"{s['by_swap']['gender']['rate']*100:9.2f}%  "
              f"{s['by_swap']['race']['rate']*100:9.2f}%")

    print("\nCross-model Jaccard (flipped-pair-set overlap):")
    for k, v in jaccard.items():
        print(f"  {k:60s}  jaccard={v['jaccard']}  inter={v['intersection']}  union={v['union']}")

    print("\nShared flips:")
    for k, v in summary["shared_flips"].items():
        print(f"  {k}: {v}")
    print(f"  total distinct flipped pairs (any model): "
          f"{summary['n_distinct_flipped_pairs_any_model']}")


if __name__ == "__main__":
    main()
