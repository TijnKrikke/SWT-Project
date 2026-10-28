import csv
import re
import time
from typing import Dict, Iterable, List, Tuple

import bert_score
import numpy as np
import torch
from config import (
    BASELINE, BERTSCORE_MODEL, BERTSCORE_NUM_LAYERS, MEASURE_LATENCY,
    QA_DATASET_PATH, RETRIEVAL_METRIC_KS,
)
from data_processing import load_qa_dataset
from generation import generate_answer
from retrieval import Retriever

_CIT_RE = re.compile(r"\[(?:Item_)?\d+\]")


def normalize_for_eval(ans: str) -> str:
    """Normalize the answer to make it easier to compare.

    Args:
        ans (str): The answer that needs to be normalized.

    Returns:
        str: The normalized answer.
    """
    a = (ans or "").strip()
    a = _CIT_RE.sub("", a)
    a = re.sub(r"\s+", " ", a)
    return a.strip()


def compute_retrieval_metrics(
    ctx_idxs: List[int],
    relevant_idxs: List[int],
    k_values: Iterable[int],
) -> Tuple[Dict[int, Tuple[float, float]], float]:
    """Compute P@K, R@K, MRR.

    Args:
        ctx_idxs (List[int]): The ids of the chunks retrieved for context.
        relevant_idxs (List[int]): Indices of the chunks that support the answer.
        k_values (Iterable[int]): P@K and R@K will be compute for every k in this list.

    Returns:
        Tuple[Dict[int, Tuple[float, float]], float]: P@K, R@K and MMR
    """

    if not relevant_idxs:
        return {k: (0.0, 0.0) for k in k_values}, 0.0

    rel_set = set(relevant_idxs)
    rel_flags = [1 if i in rel_set else 0 for i in ctx_idxs]

    metrics = {}

    for k in k_values:
        k_eff = min(k, len(rel_flags))
        topk = rel_flags[:k_eff]
        hit = sum(topk)
        precision = hit / k_eff if k_eff else 0.0
        recall = hit / len(rel_set) if rel_set else 0.0
        metrics[k] = (precision, recall)

    mrr = 0.0
    for i, rel in enumerate(rel_flags):
        if rel:
            mrr = 1.0 / (i + 1)
            break

    return metrics, mrr


def run_evaluation(
    retriever: Retriever,
    device: torch.device,
    csv_out: str = "results.csv",
) -> None:
    """Evaluate the current model using the QA dataset. Compute and store the metrics in CSV.

    Args:
        retriever (Retriever): The retriever used to retrieve chunks.
        device (torch.Device): The device used for BERTScore.
        csv_out (str, optional): The location where the csv file will be stored. Defaults to "results.csv".
    """
    qa = load_qa_dataset(QA_DATASET_PATH)
    story_to_chunk = {}
    for idx, c in enumerate(retriever.chunks):
        story_to_chunk.setdefault(c.story_id, []).append(idx)

    all_results = []
    latencies = []
    for idx, item in enumerate(qa, start=1):
        qid = item.get("id", idx)
        story_id = item.get("story_id")
        question = item["question"]
        reference = item["answer"]
        start_ts = time.perf_counter()
        answer, ctx_idxs, _ = generate_answer(question, retriever)
        latency = time.perf_counter() - start_ts
        if MEASURE_LATENCY:
            latencies.append(latency)
        if not ctx_idxs:
            if BASELINE:
                ans_eval = normalize_for_eval(answer)
                P, R, _ = bert_score.score(
                    [ans_eval], [reference], lang="nl", model_type=BERTSCORE_MODEL,
                    num_layers=BERTSCORE_NUM_LAYERS, device=str(device), verbose=False)
                bert = float(((P[0] + R[0]) / 2).item())
                # Baseline does not retrieve stories
                result = {
                    "id": qid,
                    "correct_story_id": story_id,
                    "question": question,
                    "reference": reference,
                    "answer": answer,
                    "bertscore": round(bert, 4)}
                all_results.append(result)
            continue
        ans_eval = normalize_for_eval(answer)

        P, R, _ = bert_score.score(
            [ans_eval], [reference], lang="nl", model_type=BERTSCORE_MODEL,
            num_layers=BERTSCORE_NUM_LAYERS, device=str(device), verbose=False)

        bert = float(((P[0] + R[0]) / 2).item())
        retrieval = {}
        mrr = 0.0
        story_hit = 0
        src_stories = sorted({retriever.chunks[i].story_id for i in ctx_idxs})

        if story_id is not None:
            story_hit = int(story_id in src_stories)
            relevant = story_to_chunk.get(story_id, [])
            retrieval, mrr = compute_retrieval_metrics(ctx_idxs, relevant, RETRIEVAL_METRIC_KS)

        cited_story = retriever.chunks[ctx_idxs[0]].story_id
        citation_correct = int(cited_story == story_id)
        result = {
            "id": qid,
            "correct_story_id": story_id,
            "question": question,
            "reference": reference,
            "answer": answer,
            "bertscore": round(bert, 4),
            "story_hit": story_hit,
            "citation_correct": citation_correct,
            "mrr": round(mrr, 4),
            "latency_ms": round(latency * 1000, 2),
            "retrieved_story_ids": src_stories,
        }

        for k, (p, r) in retrieval.items():
            result[f"P@{k}"] = round(p, 4)
            result[f"R@{k}"] = round(r, 4)
        all_results.append(result)

    summary = {}

    if all_results:
        summary["avg_bertscore"] = float(np.mean([r["bertscore"] for r in all_results]))
        if not BASELINE:
            summary["story_hit_rate"] = float(np.mean([r["story_hit"] for r in all_results]))
            summary["citation_accuracy"] = float(np.mean([r["citation_correct"] for r in all_results]))
            summary["mrr"] = float(np.mean([r["mrr"] for r in all_results]))

            for k in RETRIEVAL_METRIC_KS:
                summary[f"P@{k}"] = float(np.mean([r.get(f"P@{k}", 0) for r in all_results]))
                summary[f"R@{k}"] = float(np.mean([r.get(f"R@{k}", 0) for r in all_results]))

        if latencies:
            p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
            summary["latency_p50_ms"] = p50 * 1000
            summary["latency_p95_ms"] = p95 * 1000
            summary["latency_p99_ms"] = p99 * 1000

    if all_results:
        keys = sorted(all_results[0].keys())
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)

        print(summary)
