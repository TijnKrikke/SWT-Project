"""
Retrieval module for hybrid search and reranking.
Implements BM25, dense embeddings, fusion, query rewriting, and multi-stage reranking.
"""

from typing import List, Optional

import numpy as np
from config import (
    BM25_CANDIDATES,
    DENSE_CANDIDATES,
    FINAL_CONTEXT_CHUNKS,
    REFUSAL_THRESHOLD,
    RERANK_KEEP_STAGE1,
    USE_HYBRID,
    USE_RERANK,
)
from data_processing import Chunk, bm25_tokenize, buildVectorDB
from sentence_transformers import CrossEncoder, SentenceTransformer


class Retriever:
    """Hybrid retriever combining BM25, dense embeddings and reranking."""

    def __init__(
        self,
        chunks: List[Chunk],
        embedder: SentenceTransformer,
        reranker_stage1: CrossEncoder,
        reranker_stage2: Optional[CrossEncoder],
    ) -> None:
        """
        Initialize retriever with chunks and models.

        Args:
            chunks: List of Chunk objects from the corpus.
            embedder: Sentence transformer for dense embeddings.
            reranker_stage1: First-stage cross-encoder reranker.
            reranker_stage2: Optional second-stage cross-encoder reranker.
        """
        self.chunks = chunks
        self.embedder = embedder
        self.rer1 = reranker_stage1
        self.rer2 = reranker_stage2

        self.texts = [c.text for c in chunks]

        # BM25
        self.bm25_tokens = [bm25_tokenize(t) for t in self.texts]
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi(self.bm25_tokens)

        self.store = buildVectorDB(chunks, self.embedder)

    def bm25_search(self, query: str, k: int) -> List[int]:
        """BM25 search for top k chunks.

        Args:
            query (str): The query for which chunks need to be searched.
            k (int): The number of chunks to return.

        Returns:
            List[int]: The top-k chunks retrieved by BM25.
        """
        qt = bm25_tokenize(query)
        scores = self.bm25.get_scores(qt)
        idx = np.argsort(scores)[::-1][:k]
        return idx.tolist()

    def dense_search(self, query: str, k: int, refusal_threshold: float = REFUSAL_THRESHOLD) -> List[int]:
        """Dense embedding search for top k chunks.

        Args:
            query (str): The query for which to find close neighbors.
            k (int): The number of neighbors to retrieve.
            refusal_threshold (float, optional): If the average top k similarity score is
                are below this threshold, we do not return any chunks. Defaults to REFUSAL_THRESHOLD.

        Returns:
            List[int]: The closes k neighbors.
        """
        qemb = self.embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        results = self.store.query(qemb.tolist(), k)
        # threshold for retrieved chunks
        scores = 1 - np.array(results["distances"][0], dtype=np.float32)
        if scores[:5].mean() < refusal_threshold:
            print(scores[:5], scores[:5].mean())
            return None
        return [int(idx) for idx in results["ids"][0]]

    def fuse(self, a: List[int], b: List[int]) -> List[int]:
        """Fuse two ranked lists while preserving order and removing duplicates.

        Args:
            a (List[int]): The first ranked list.
            b (List[int]): The second ranked list.

        Returns:
            List[int]: The fused list without duplicates.
        """
        seen = set()
        out = []
        for i in a + b:
            if i in seen:
                continue
            seen.add(i)
            out.append(i)
        return out

    def rerank(self, query: str, cand_idx: List[int], keep_stage1: int) -> List[int]:
        """
        Multi-stage reranking: stage1 reduces candidates, optional stage2 further reorders.

        Args:
            query: Original question.
            cand_idx: Candidate chunk indices to rerank.
            keep_stage1: Number of candidates to keep after stage 1.

        Returns:
            list[int]: Reranked chunk indices.
        """
        if not cand_idx:
            return []

        cand_texts = [self.texts[i] for i in cand_idx]
        pairs = [(query, t) for t in cand_texts]

        s1 = self.rer1.predict(pairs)
        order1 = np.argsort(-np.array(s1))[:keep_stage1]
        stage1 = [cand_idx[i] for i in order1]

        if self.rer2 is None:
            return stage1

        cand2_texts = [self.texts[i] for i in stage1]
        pairs2 = [(query, t) for t in cand2_texts]
        s2 = self.rer2.predict(pairs2)
        order2 = np.argsort(-np.array(s2))
        return [stage1[i] for i in order2]

    def retrieve(
        self,
        question: str,
        *,
        bm25_k: int = BM25_CANDIDATES,
        dense_k: int = DENSE_CANDIDATES,
        keep_stage1: int = RERANK_KEEP_STAGE1,
        final_k: int = FINAL_CONTEXT_CHUNKS,
        use_hybrid: bool = USE_HYBRID,
        use_rerank: bool = USE_RERANK,
    ) -> List[int]:
        """Implements the full retrieval pipeline with ChromaDB + BM25 + Reranking.

        Args:
            question (str): The question for which chunks need to be found.
            bm25_k (int, optional): The number of candidates to return for BM25. Defaults to BM25_CANDIDATES.
            dense_k (int, optional): The number of candidates to return for ChromaDB. Defaults to DENSE_CANDIDATES.
            keep_stage1 (int, optional): The number of candidates to keep after the first reranker.
                Defaults to RERANK_KEEP_STAGE1.
            final_k (int, optional): The final top-k chunks. Defaults to FINAL_CONTEXT_CHUNKS.
            use_hybrid (bool, optional): Whether to use BM25 to retrieve chunks in combination with ChromaDB.
                Defaults to USE_HYBRID.
            use_rerank (bool, optional): Wether to rerank the retrieved chunks. Defaults to USE_RERANK.

        Returns:
            List[int]: The top-k chunks.
        """

        ds_chunks = self.dense_search(question, dense_k)
        if not ds_chunks:
            return None

        if use_hybrid:
            bm_chunks = self.bm25_search(question, bm25_k)
            fused = self.fuse(bm_chunks, ds_chunks)

        if use_rerank:
            if not use_hybrid:
                fused = ds_chunks
            reranked = self.rerank(question, fused, keep_stage1=keep_stage1)
        else:
            if use_hybrid:
                reranked = fused
            else:
                reranked = ds_chunks
        return reranked[:final_k]
