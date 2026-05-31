"""E4 aggregation: per-model blind-judge mention rates on flipped pairs.

For each judged_e1_*_reason.jsonl, compute:
  - per-swap-type mention rates for race / gender / age
  - feature_overlap distribution
  - the headline question: when a pair flipped on demographic axis X, did
    EITHER paired reason mention axis X? (vs the older "did the reason
    mention ANY demographic field" - too coarse)

Output: prints a table, writes JSON summary.
"""
import argparse, json, glob, sys
from collections import Counter, defaultdict


def dedup(rows):
    """Last-write-wins dedup by pair key."""
    def k(r):
        return (r.get('swap'), r.get('qid'),
                str(r.get('a_val')), str(r.get('b_val')),
                r.get('gender'), r.get('race'), r.get('age'))
    ix = {}
    for r in rows:
        ix[k(r)] = r
    return list(ix.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="results/e4_summary.json")
    args = ap.parse_args()

    summary = {}
    print(f"\n{'model':25s} {'swap':7s} {'n flips':>8s}  "
          f"{'mention(this axis)':>20s}  {'mention(any axis)':>18s}  {'overlap=same':>13s}")

    for path in sorted(args.inputs):
        rows = dedup([json.loads(l) for l in open(path)])
        # Identify model from the source pair file by inspecting one a_cot
        # carrier OR by filename.
        label = path.split('judged_e1_')[-1].split('_reason')[0]
        labeled = [r for r in rows if r.get('mentions_race') is not None]

        per_swap = {}
        for sw in ("age", "gender", "race"):
            sub = [r for r in labeled if r.get('swap') == sw]
            n = len(sub)
            if n == 0:
                continue
            # Mention rate for the swapped axis (the relevant one)
            mention_this_axis = sum(
                1 for r in sub if r.get(f'mentions_{sw}') == 'yes'
            ) / n
            # Mention rate for ANY demographic axis
            mention_any = sum(
                1 for r in sub
                if 'yes' in (r.get('mentions_race'), r.get('mentions_gender'), r.get('mentions_age'))
            ) / n
            # Feature overlap distribution
            overlap = Counter(r.get('feature_overlap') for r in sub)
            same_pct = (overlap.get('same', 0) / n) if n else None
            per_swap[sw] = {
                "n_flips": n,
                "mention_this_axis": round(mention_this_axis, 4),
                "mention_any_axis":  round(mention_any, 4),
                "overlap_same":      round(same_pct, 4),
                "overlap_partial":   round(overlap.get('partial', 0) / n, 4),
                "overlap_different": round(overlap.get('different', 0) / n, 4),
            }
            print(f"  {label:25s} {sw:7s} {n:8d}  "
                  f"{mention_this_axis*100:18.2f}%  "
                  f"{mention_any*100:16.2f}%  "
                  f"{same_pct*100:12.2f}%")

        # Overall row
        total_n = sum(per_swap[sw]['n_flips'] for sw in per_swap)
        if total_n > 0:
            tot_mention_axis = sum(
                per_swap[sw]['mention_this_axis'] * per_swap[sw]['n_flips']
                for sw in per_swap
            ) / total_n
            tot_mention_any  = sum(
                per_swap[sw]['mention_any_axis'] * per_swap[sw]['n_flips']
                for sw in per_swap
            ) / total_n
            tot_overlap_same = sum(
                per_swap[sw]['overlap_same'] * per_swap[sw]['n_flips']
                for sw in per_swap
            ) / total_n
            print(f"  {label:25s} {'TOTAL':7s} {total_n:8d}  "
                  f"{tot_mention_axis*100:18.2f}%  "
                  f"{tot_mention_any*100:16.2f}%  "
                  f"{tot_overlap_same*100:12.2f}%")

        summary[label] = {
            "source": path,
            "n_judged": len(labeled),
            "n_total_pairs_in_file": len(rows),
            "per_swap": per_swap,
            "overall": {
                "mention_this_axis": round(tot_mention_axis, 4) if total_n else None,
                "mention_any_axis":  round(tot_mention_any, 4)  if total_n else None,
                "overlap_same":      round(tot_overlap_same, 4) if total_n else None,
            },
        }

    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {args.out}")


if __name__ == "__main__":
    main()
