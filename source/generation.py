from ollama import chat

from config import BASELINE, GENERATION_MODEL, MAX_NEW_TOKENS
from retrieval import Retriever


def _generate(messages: list[dict]) -> str:
    response = chat(
        model=GENERATION_MODEL,
        messages=messages,
        stream=False,
        think=False,
        options={"num_predict": MAX_NEW_TOKENS, "temperature": 0},
    )
    return response["message"]["content"].strip()


def generate_answer(
    question: str,
    retriever: Retriever,
) -> tuple[str, list[int] | None, list[dict]]:
    """The model that generates the final output with or without conversational context.

    Args:
        question (str): The user's question.
        retriever (Retriever): The retriever used to retrieve chunks.

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

        answer = _generate(messages)
        return answer, None, messages

    # with RAG
    ctx_idxs = retriever.retrieve(question)

    if not ctx_idxs:
        raise ValueError("No context chunks found for the given question.")

    context_lines = []
    for idx in ctx_idxs:
        chunk = retriever.chunks[idx]
        meta_parts = []
        if chunk.metadata:
            for k, v in chunk.metadata.items():
                val = ", ".join(str(x) for x in v) if isinstance(v, list) else str(v)
                if val:
                    meta_parts.append(f"{k.replace('_', ' ').capitalize()}: {val}")
        meta_str = f"Metadata: {' | '.join(meta_parts)}\n" if meta_parts else ""
        context_lines.append(f"[{chunk.story_id}] {chunk.title}\n{meta_str}{chunk.text}")

    context = "\n\n".join(context_lines)

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

    answer = _generate(messages)

    cite_story = retriever.chunks[ctx_idxs[0]].story_id
    answer += f"[{cite_story}]"

    print("PROMPT:")
    print(question)

    return answer, ctx_idxs, messages
