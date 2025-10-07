import json
import argparse
from typing import Dict, List
import numpy as np


class LALCalculator:
    """
    Minimal calculator for LAL and LAL normalized by logit range.
    Extracted from new_metrics.py with only the necessary components.
    """

    @staticmethod
    def compute_advantage(logits: Dict[str, float], correct_answer: str) -> Dict[str, float]:
        """
        Compute max-based advantage and supporting quantities for one question.

        Returns a dict with:
          - advantage: ℓ(o*) - max_{o≠o*} ℓ(o)
          - logit_span: max(all_logits) - min(all_logits)
        """
        logit_correct = float(logits[correct_answer])
        logits_incorrect = [float(v) for k, v in logits.items() if k != correct_answer]

        max_incorrect = max(logits_incorrect)
        advantage = logit_correct - max_incorrect

        all_logits = list(map(float, logits.values()))
        logit_span = max(all_logits) - min(all_logits)

        return {
            "advantage": advantage,
            "logit_span": logit_span,
        }

    @staticmethod
    def compute_lal(adv_i: Dict[str, float], adv_j: Dict[str, float]) -> Dict[str, float]:
        """
        Compute LAL and LAL normalized by range between two levels i and j.
        """
        A_i = adv_i["advantage"]
        A_j = adv_j["advantage"]

        lal = A_i - A_j

        span_i = adv_i.get("logit_span", None)
        span_j = adv_j.get("logit_span", None)

        A_i_norm_range = (A_i / span_i) if (span_i is not None and span_i != 0) else np.nan
        A_j_norm_range = (A_j / span_j) if (span_j is not None and span_j != 0) else np.nan
        lal_norm_range = A_i_norm_range - A_j_norm_range

        return {
            "lal": lal,
            "lal_norm_range": lal_norm_range,
        }


def calculate_group_lal(group_data: List[Dict]) -> Dict:
    """
    Calculate LAL and range-normalized LAL for a group (levels 1,2,3).
    Expects a list of exactly 3 entries with keys 'case' and 'result'.
    """
    # Sort by level and index by level
    sorted_data = sorted(group_data, key=lambda x: x["case"]["level"])
    levels = {}
    for data in sorted_data:
        level = data["case"]["level"]
        res = data["result"]
        levels[level] = {
            "logits": res["option_logits"],
            "correct": data["case"]["answer"],
        }

    # Compute per-level advantages
    adv = {}
    for level in [1, 2, 3]:
        if level in levels:
            adv[level] = LALCalculator.compute_advantage(
                levels[level]["logits"], levels[level]["correct"]
            )

    # Prepare output
    out = {
        "set_id": sorted_data[0]["case"]["set_id"],
        "group_id": sorted_data[0]["case"]["id"].split("_level")[0],
        "lal_1_to_2": None,
        "lal_2_to_3": None,
        "lal_1_to_3": None,
        "lal_norm_1_to_2": None,
        "lal_norm_2_to_3": None,
        "lal_norm_1_to_3": None,
    }

    transitions = [(1, 2), (2, 3), (1, 3)]
    for i, j in transitions:
        if i in adv and j in adv:
            res = LALCalculator.compute_lal(adv[i], adv[j])
            out[f"lal_{i}_to_{j}"] = res["lal"]
            out[f"lal_norm_{i}_to_{j}"] = res["lal_norm_range"]

    return out


def process_json_file(input_path: str, output_path: str = None) -> List[Dict]:
    """
    Process an input JSON file and compute LAL (and range-normalized) for all groups.
    If output_path is provided, writes a JSON file with the results list.
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    # Group items by set_id
    groups = {}
    for item in data:
        set_id = item["case"]["set_id"]
        groups.setdefault(set_id, []).append(item)

    results: List[Dict] = []
    failed: List[Dict] = []

    for set_id, group_items in groups.items():
        if len(group_items) != 3:
            failed.append({"set_id": set_id, "error": f"Incomplete group: {len(group_items)} items"})
            continue
        try:
            results.append(calculate_group_lal(group_items))
        except Exception as e:
            failed.append({"set_id": set_id, "error": str(e)})

    if output_path:
        payload = {
            "metadata": {
                "input_file": input_path,
                "total_groups": len(groups),
                "processed": len(results),
                "failed": len(failed),
            },
            "results": results,
            "failed_groups": failed,
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute LAL and range-normalized LAL from results JSON")
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("--output", help="Optional path to save results JSON", default=None)
    args = parser.parse_args()

    results = process_json_file(args.input, args.output)

    # Quick textual summary for 1→3 transition (common focus)
    lal_vals = [r["lal_1_to_3"] for r in results if r.get("lal_1_to_3") is not None]
    lal_nr_vals = [r["lal_norm_1_to_3"] for r in results if r.get("lal_norm_range_1_to_3") is not None]

    if lal_vals:
        arr = np.array([v for v in lal_vals if v is not None and not np.isnan(v)])
        print(f"LAL 1→3: n={arr.size}, mean={arr.mean():.4f}, std={arr.std(ddof=1) if arr.size>1 else 0:.4f}")
    if lal_nr_vals:
        arr = np.array([v for v in lal_nr_vals if v is not None and not np.isnan(v)])
        print(f"LAL_norm_range 1→3: n={arr.size}, mean={arr.mean():.4f}, std={arr.std(ddof=1) if arr.size>1 else 0:.4f}")


if __name__ == "__main__":
    main()


