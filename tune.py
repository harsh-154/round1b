# tune.py  ─ quick hyper-parameter search (≈30 s on CPU)
import json, subprocess, itertools, os, sys, statistics

COLLS = [
    "sample_data/input/collection1",
    "sample_data/input/collection2",
    "sample_data/input/collection3",
]

GRID = {
    "max_per_doc": [1, 2, 3],      # 1 = no diversity cap
    "top_n":       [3, 5, 7],
    "kw_boost":    [0.0, 0.05, 0.10],
}

def run_cmd(cmd):
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return out.stdout

best = None
for max_per_doc, top_n, kw_boost in itertools.product(*GRID.values()):
    os.environ["MAX_PER_DOC"] = str(max_per_doc)
    os.environ["TOP_N"]       = str(top_n)
    os.environ["KW_BOOST"]    = str(kw_boost)

    prec, bleu = [], []
    for c in COLLS:
        # inference
        run_cmd(f"python main.py {c}")

        # paths
        pred = os.path.join(c, "challenge1b_output.json")
        gt   = os.path.join(c, "challenge1b_output.gt.json")

        # evaluate
        res = run_cmd(f"python evaluate.py --pred {pred} --gt {gt}")
        p = float(res.splitlines()[0].split()[-1])
        b = float(res.splitlines()[2].split()[-1])
        prec.append(p); bleu.append(b)

    score = statistics.mean(prec) * 0.6 + statistics.mean(bleu) * 0.4
    print(f"{max_per_doc=}, {top_n=}, {kw_boost=}: "
          f"P@5={statistics.mean(prec):.2f}, BLEU={statistics.mean(bleu):.2f}, "
          f"combined={score:.3f}")

    if best is None or score > best[0]:
        best = (score, max_per_doc, top_n, kw_boost)

print("\nBEST →", best)