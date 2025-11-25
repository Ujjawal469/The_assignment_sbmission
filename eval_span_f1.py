import json
import argparse
from collections import defaultdict
from labels import label_is_pii


def load_gold(path):
    gold = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            uid = obj["id"]
            spans = []
            for e in obj.get("entities", []):
                spans.append((e["start"], e["end"], e["label"]))
            gold[uid] = spans
    return gold


def load_pred(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    pred = {}
    for uid, ents in obj.items():
        spans = []
        for e in ents:
            spans.append((e["start"], e["end"], e["label"]))
        pred[uid] = spans
    return pred


def compute_prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    gold = load_gold(args.gold)
    pred = load_pred(args.pred)

    labels = set()
    for spans in gold.values():
        for _, _, lab in spans:
            labels.add(lab)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for uid in gold.keys():
        g_spans = set(gold.get(uid, []))
        p_spans = set(pred.get(uid, []))

        for span in p_spans:
            if span in g_spans:
                tp[span[2]] += 1
            else:
                fp[span[2]] += 1
        for span in g_spans:
            if span not in p_spans:
                fn[span[2]] += 1

    print("Per-entity metrics:")
    macro_f1_sum = 0.0
    macro_count = 0

    for lab in sorted(labels):
        p, r, f1 = compute_prf(tp[lab], fp[lab], fn[lab])
        print(f"{lab:15s} P={p:.3f} R={r:.3f} F1={f1:.3f}")
        macro_f1_sum += f1
        macro_count += 1

    macro_f1 = macro_f1_sum / max(1, macro_count)
    print(f"\nMacro-F1: {macro_f1:.3f}")

    pii_tp = pii_fp = pii_fn = 0
    non_tp = non_fp = non_fn = 0

    for uid in gold.keys():
        g_spans = gold.get(uid, [])
        p_spans = pred.get(uid, [])

        g_pii = set((s, e, "PII") for s, e, lab in g_spans if label_is_pii(lab))
        g_non = set((s, e, "NON") for s, e, lab in g_spans if not label_is_pii(lab))
        p_pii = set((s, e, "PII") for s, e, lab in p_spans if label_is_pii(lab))
        p_non = set((s, e, "NON") for s, e, lab in p_spans if not label_is_pii(lab))

        for span in p_pii:
            if span in g_pii:
                pii_tp += 1
            else:
                pii_fp += 1
        for span in g_pii:
            if span not in p_pii:
                pii_fn += 1

        for span in p_non:
            if span in g_non:
                non_tp += 1
            else:
                non_fp += 1
        for span in g_non:
            if span not in p_non:
                non_fn += 1

    p, r, f1 = compute_prf(pii_tp, pii_fp, pii_fn)
    print(f"\nPII-only metrics: P={p:.3f} R={r:.3f} F1={f1:.3f}")
    p2, r2, f12 = compute_prf(non_tp, non_fp, non_fn)
    print(f"Non-PII metrics: P={p2:.3f} R={r2:.3f} F1={f12:.3f}")


if __name__ == "__main__":
    main()
