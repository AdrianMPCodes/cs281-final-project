"""Enumerate counterfactual pairs and detect flips."""
import json, sys, argparse
from collections import defaultdict, Counter
from itertools import combinations

AGES = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
GENDERS = ["female", "male", "non-binary"]
RACES = ["white", "Black", "Asian", "Hispanic", "Native American"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.decisions)]
    idx = {(r["qid"], r["age"], r["gender"], r["race"]): r for r in rows}
    qids = sorted({r["qid"] for r in rows})

    pairs = []
    # age pairs: hold (qid, gender, race), vary age
    for qid in qids:
        for g in GENDERS:
            for r_ in RACES:
                for a1, a2 in combinations(AGES, 2):
                    A = idx[(qid, a1, g, r_)]
                    B = idx[(qid, a2, g, r_)]
                    pairs.append({"swap": "age", "qid": qid, "gender": g, "race": r_,
                                  "a_val": a1, "b_val": a2,
                                  "a_decision": A["decision"], "b_decision": B["decision"],
                                  "a_reason": A["reason"], "b_reason": B["reason"]})
    # gender pairs
    for qid in qids:
        for r_ in RACES:
            for a in AGES:
                for g1, g2 in combinations(GENDERS, 2):
                    A = idx[(qid, a, g1, r_)]
                    B = idx[(qid, a, g2, r_)]
                    pairs.append({"swap": "gender", "qid": qid, "race": r_, "age": a,
                                  "a_val": g1, "b_val": g2,
                                  "a_decision": A["decision"], "b_decision": B["decision"],
                                  "a_reason": A["reason"], "b_reason": B["reason"]})
    # race pairs
    for qid in qids:
        for g in GENDERS:
            for a in AGES:
                for r1, r2 in combinations(RACES, 2):
                    A = idx[(qid, a, g, r1)]
                    B = idx[(qid, a, g, r2)]
                    pairs.append({"swap": "race", "qid": qid, "gender": g, "age": a,
                                  "a_val": r1, "b_val": r2,
                                  "a_decision": A["decision"], "b_decision": B["decision"],
                                  "a_reason": A["reason"], "b_reason": B["reason"]})

    flips = 0
    by_swap = Counter()
    by_swap_total = Counter()
    for p in pairs:
        by_swap_total[p["swap"]] += 1
        p["flipped"] = (p["a_decision"] is not None and p["b_decision"] is not None
                        and p["a_decision"] != p["b_decision"])
        if p["flipped"]:
            flips += 1
            by_swap[p["swap"]] += 1

    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"Total pairs: {len(pairs)}")
    print(f"Total flips: {flips}  ({flips/len(pairs)*100:.2f}%)")
    print("Flips by swap type:")
    for s in ["age", "gender", "race"]:
        n = by_swap[s]; tot = by_swap_total[s]
        print(f"  {s:8s}: {n:4d} / {tot:5d}  ({n/tot*100:.2f}%)")


if __name__ == "__main__":
    main()
