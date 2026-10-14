"""
Evaluation module for system assessment.
Computes BLEU, BERTScore, retrieval metrics, citation accuracy, and dialogue coherence.
"""

import re

import torch
from config import BEAM_SIZE, MAX_NEW_TOKENS, BASELINE
from retrieval import Retriever
from transformers import AutoModelForCausalLM, AutoTokenizer


def strip_leading_enum(text: str) -> str:
    return re.sub(r"^\s*\(?[ivxlcdm]+\)?\s*", "", text, flags=re.I)


@torch.no_grad()
def generate_answer(
    question: str,
    retriever: Retriever,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
) -> tuple[str, list[int], list[dict]]:
    """The model that generates the final output with or without conversational context.

    Args:
        question (str): The user's question.
        retriever (Retriever): The retriever used to retrieve chunks.
        tokenizer (AutoTokenizer): The tokenizer used to tokenize the chunks for the model.
        model (AutoModelForCausalLM): The model used to generate the answers.
        device (torch.device): The device on which the model is stored.

    Returns:
        tuple[str, list[int], list[dict]]: The generated answer, the used context chunk ids,
            the message used for generation.
    """
    if BASELINE: # no RAG
        messages = [
            {
                "role": "user",
                "content": (question)
            }
        ]

        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)

        outputs = model.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=BEAM_SIZE,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs.shape[1] :]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return answer, None, messages

    # with RAG
    ctx_idxs = retriever.retrieve(question)

    if not ctx_idxs:
        raise ValueError("No context chunks found for the given question.")

    context_lines = []
    for i, idx in enumerate(ctx_idxs, start=1):
        chunk = retriever.chunks[idx]
        clean_text = strip_leading_enum(chunk.text)
        context_lines.append(f"[{i}] {clean_text}")

    context = "\n".join(context_lines)

    # maybe make this in Dutch later (content)
    messages = [
        {
            "role": "user",
            "content": (
                "Answer the question using the context below in one sentence.\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{question}"
            ),
        }
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)

    outputs = model.generate(
        inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=BEAM_SIZE,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = outputs[0][inputs.shape[1] :]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    cite_page = retriever.chunks[ctx_idxs[0]].page # probably redundant, maybe adapt, depends
    answer += f"[{cite_page}]"

    print("PROMPT:")
    print(question)

    return answer, ctx_idxs, messages
