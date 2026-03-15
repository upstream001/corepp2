#!/usr/bin/env python3
import argparse
import csv
import json


def read_gt(gt_csv):
    gt = {}
    with open(gt_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = (row.get("filename") or "").strip()
            vv = (row.get("volume_ml") or "").strip()
            if fn and vv:
                gt[fn] = float(vv)
    return gt


def main():
    parser = argparse.ArgumentParser(description="Compare predicted volume against GT using mapping.json.")
    parser.add_argument("--mapping", default="/home/tianqi/corepp2/data/20260312_dataset/mapping.json", help="mapping.json path")
    parser.add_argument("--ground-truth", default="/home/tianqi/corepp2/ground_truth.csv", help="ground_truth.csv path")
    parser.add_argument("--pred", default="/home/tianqi/corepp2/shape_completion_results.csv", help="shape_completion_results.csv path")
    args = parser.parse_args()

    mapping = json.load(open(args.mapping, "r", encoding="utf-8"))
    gt = read_gt(args.ground_truth)

    matched = []
    with open(args.pred, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame_id = (row.get("frame_id") or "").strip()
            pred_v = (row.get("mesh_volume_ml") or "").strip()
            if not frame_id or not pred_v:
                continue
            k = frame_id if frame_id.endswith(".ply") else frame_id + ".ply"
            full = mapping.get(k)
            if not full or full not in gt:
                continue
            p = float(pred_v)
            g = gt[full]
            ape = abs(p - g) / g * 100 if g > 0 else 0.0
            matched.append((frame_id, full, p, g, ape, p - g))

    if not matched:
        print("No matched rows found.")
        return

    mape = sum(x[4] for x in matched) / len(matched)
    mae = sum(abs(x[5]) for x in matched) / len(matched)
    bias = sum(x[5] for x in matched) / len(matched)

    print(f"Matched rows: {len(matched)}")
    print(f"MAPE (%): {mape:.3f}")
    print(f"MAE (mL): {mae:.3f}")
    print(f"Bias (mL): {bias:.3f}")


if __name__ == "__main__":
    main()
