import csv
import re
import time
from typing import Dict, Iterable, List, Tuple

import bert_score
import numpy as np
import sacrebleu
import torch
from config import MEASURE_LATENCY, QA_DATASET_PATH, RETRIEVAL_METRIC_KS, BASELINE
from data_processing import load_qa_dataset
from generation import generate_answer
from retrieval import Retriever
from transformers import AutoModelForCausalLM, AutoTokenizer

_CIT_RE = re.compile(r"\[\d+\]")


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
        relevant_idxs (List[int]): The page numbers where the answer should have come from.
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
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    csv_out: str = "results.csv",
) -> None:
    """Evaluate the current model using the QA dataset. Compute and store the metrics in CSV.

    Args:
        retriever (Retriever): The retriever used to retrieve chunks.
        tokenizer (AutoTokenize): The tokenizer to tokenize chunks for the model.
        model (AutoModelForCausalLM): The model used to generate output.
        device (torch.Device): The device used for the model.
        csv_out (str, optional): The location where the csv file will be stored. Defaults to "results.csv".
    """
    qa = load_qa_dataset(QA_DATASET_PATH)
    page_to_chunk = {}
    for idx, c in enumerate(retriever.chunks):
        page_to_chunk.setdefault(c.page, []).append(idx)

    all_results = []
    latencies = []
    for idx, item in enumerate(qa, start=1):
        qid = item.get("id", idx)
        page = item.get("page")
        question = item["question"]
        reference = item["answer"]
        start_ts = time.perf_counter()
        answer, ctx_idxs, _ = generate_answer(question, retriever, tokenizer, model, device)
        latency = time.perf_counter() - start_ts
        if MEASURE_LATENCY:
            latencies.append(latency)
        if not ctx_idxs:
            if BASELINE:
                ans_eval = normalize_for_eval(answer)
                bleu = sacrebleu.corpus_bleu([ans_eval], [[reference]]).score
                P, R, _ = bert_score.score([ans_eval], [reference], lang="en", model_type="bert-base-uncased", verbose=False)
                bert = float(((P[0] + R[0]) / 2).item())
                # Baseline does not retrieve pages
                result = {
                    "id": qid,
                    "question": question,
                    "reference": reference,
                    "answer": answer,
                    "bleu": round(bleu, 3),
                    "bertscore": round(bert, 4)}
                all_results.append(result)
            continue
        ans_eval = normalize_for_eval(answer)
        bleu = sacrebleu.corpus_bleu([ans_eval], [[reference]]).score

        P, R, _ = bert_score.score([ans_eval], [reference], lang="en", model_type="bert-base-uncased", verbose=False)

        bert = float(((P[0] + R[0]) / 2).item())
        retrieval = {}
        mrr = 0.0
        page_hit = 0
        src_pages = sorted({retriever.chunks[i].page for i in ctx_idxs})

        if page is not None:
            page_hit = int(page in src_pages)
            if page in src_pages:
                page_hit = 1.0
            relevant = page_to_chunk.get(page, [])
            retrieval, mrr = compute_retrieval_metrics(ctx_idxs, relevant, RETRIEVAL_METRIC_KS)

        cited_page = retriever.chunks[ctx_idxs[0]].page
        citation_correct = int(cited_page == page)
        result = {
            "id": qid,
            "question": question,
            "reference": reference,
            "answer": answer,
            "bleu": round(bleu, 3),
            "bertscore": round(bert, 4),
            "page_hit": page_hit,
            "citation_correct": citation_correct,
            "mrr": round(mrr, 4),
            "latency_ms": round(latency * 1000, 2),
            "retrieved_pages": src_pages,
        }

        for k, (p, r) in retrieval.items():
            result[f"P@{k}"] = round(p, 4)
            result[f"R@{k}"] = round(r, 4)
        all_results.append(result)

    summary = {}

    if all_results:
        summary["avg_bleu"] = float(np.mean([r["bleu"] for r in all_results]))
        summary["avg_bertscore"] = float(np.mean([r["bertscore"] for r in all_results]))
        if not BASELINE:
            summary["page_hit_rate"] = float(np.mean([r["page_hit"] for r in all_results]))
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
