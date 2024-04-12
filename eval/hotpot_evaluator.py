import argparse
import json
import re
import string
from collections import Counter

from datasets import load_dataset


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--pred_path", type=str, help="Path to the predictions")
    args = parser.parse_args()

    dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")

    with open(args.pred_path, "r", encoding="utf-8") as f:
        preds_dict = json.load(f)

    preds, answers = [], []
    for x in dataset:
        if x["id"] in preds_dict:
            preds.append(preds_dict[x["id"]].split(".")[0])
            answers.append(x["answer"])
        else:
            print(f"Missing {x['id']}")

    total_score = 0.0
    for pred, ans in zip(preds, answers):
        total_score += qa_f1_score(pred, ans)

    print(f"{args.pred_path}'s F1 score: {round(100 * total_score / len(preds), 2)}")
