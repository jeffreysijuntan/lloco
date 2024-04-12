import argparse
import io
import json


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def load_jsonl(f, mode="r"):
    """Load a .jsonl file into a list of dictionaries."""
    f = _make_r_io_base(f, mode)
    jlist = [json.loads(line) for line in f]
    f.close()
    return jlist


def preprocess_quality(quality_path):
    data = load_jsonl(quality_path)
    res = {}
    for entry in data:
        for q in entry["questions"]:
            qid = q["question_unique_id"]
            gold_label = q["gold_label"]
            gt = gold_label - 1
            choice = ["A", "B", "C", "D"][gt]
            res[qid] = choice
    return res


def compare(gt, pred):
    correct = 0
    for key in gt.keys():
        if key not in pred:
            raise ValueError(f"Missing {key} in predictions")
        if len(pred[key]) == 0:
            continue
        if gt[key][0] == pred[key][0]:
            correct += 1
    print("Total questions:", len(gt))
    print("Correct:", correct)
    print("Accuracy:", correct / len(gt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare ground truth and predictions."
    )
    parser.add_argument("--quality_path", type=str, help="Path to the QuALITY dataset file")
    parser.add_argument("--pred_path", type=str, help="Path to the predictions file")
    args = parser.parse_args()

    gt = preprocess_quality(args.quality_path)
    with open(args.pred_path) as f:
        pred = json.load(f)
    compare(gt, pred)
