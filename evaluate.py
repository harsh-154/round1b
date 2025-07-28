import json
import argparse
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
sm = SmoothingFunction().method4
"""evaluate.py
Quick local accuracy check.

Usage:
  python evaluate.py --pred sample_data/input/collection1/challenge1b_output.json \
                     --gt   sample_data/input/collection1/challenge1b_output.json
The script prints:
  • Precision@K for extracted sections (default K=5)
  • Mean Reciprocal Rank (MRR)
  • Average BLEU for refined_text vs ground-truth refined_text (if present)

Ground truth must list the *desired* order in `extracted_sections`.
"""


def load_pairs(path: Path) -> List[Tuple[str, int]]:
    data = json.loads(path.read_text("utf-8"))
    return [(sec["document"], sec["page_number"]) for sec in data["extracted_sections"]]


def precision_at_k(pred: List[Tuple[str, int]], gt: List[Tuple[str, int]], k: int) -> float:
    pred_k = pred[:k]
    hits = sum(1 for p in pred_k if p in gt)
    return hits / k


def mrr(pred: List[Tuple[str, int]], gt: List[Tuple[str, int]]) -> float:
    for idx, p in enumerate(pred, 1):
        if p in gt:
            return 1 / idx
    return 0.0


def average_bleu(pred_file: Path, gt_file: Path) -> float:
    def _load_subsections(path: Path):
        data = json.loads(path.read_text("utf-8"))
        return data.get("sub_section_analysis") or data.get("subsection_analysis") or []

    p_data = _load_subsections(pred_file)
    g_data = _load_subsections(gt_file)
    # map by (doc,page)
    g_map = {(s["document"], s["page_number"]): s["refined_text"] for s in g_data}
    scores = []
    for s in p_data:
        ref = g_map.get((s["document"], s["page_number"]))
        if ref:
            scores.append(sentence_bleu([ref.split()], s["refined_text"].split(), weights=(0.5, 0.5),smoothing_function=sm))
    return sum(scores) / max(len(scores), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to model output JSON")
    parser.add_argument("--gt", required=True, help="Path to ground-truth JSON")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    pred_pairs = load_pairs(Path(args.pred))
    gt_pairs = load_pairs(Path(args.gt))

    p_at_k = precision_at_k(pred_pairs, gt_pairs, args.k)
    r_mrr = mrr(pred_pairs, gt_pairs)
    bleu = average_bleu(Path(args.pred), Path(args.gt))

    print(f"Precision@{args.k}: {p_at_k:.2f}")
    print(f"MRR:             {r_mrr:.2f}")
    print(f"Avg BLEU (sub-sections): {bleu:.2f}") 