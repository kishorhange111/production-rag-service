"""Evaluate the pipeline on SQuAD 2.0: retrieval ablations, answer quality, faithfulness, abstention.

Why SQuAD 2.0? Every question comes with its gold paragraph (so retrieval is
measurable), and a third of the questions are *unanswerable*: written to look
answerable from the paragraph, but not. That makes it a direct test of whether
the system says "I don't know" instead of hallucinating.

Setup:
  corpus     every unique paragraph of the validation split (1,204 paragraphs, 35 Wikipedia articles)
  questions  a seeded sample of N answerable + N unanswerable questions

Usage (GPU):
  python -m ragservice.eval.runner retrieval --out results/retrieval.json
  python -m ragservice.eval.runner answers --generator Qwen/Qwen2.5-7B-Instruct --judge Qwen/Qwen2.5-14B-Instruct --out results/answers_7b.json
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ragservice.chunking import Document
from ragservice.eval.metrics import (
    FAITHFULNESS_PROMPT,
    abstention_scores,
    answer_contains_reference,
    exact_match,
    f1,
    ndcg_at_k,
    parse_faithfulness,
    recall_at_k,
    reciprocal_rank,
)
from ragservice.generation import build_prompt, is_abstention, parse_citations


@dataclass
class QAItem:
    question: str
    answers: list[str]  # empty for unanswerable questions
    gold_doc: str  # id of the paragraph the question was written about

    @property
    def answerable(self) -> bool:
        return bool(self.answers)


def load_squad_v2(n_per_type: int, seed: int = 0) -> tuple[list[Document], list[QAItem]]:
    """Corpus = unique validation paragraphs; questions = balanced answerable/unanswerable sample."""
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    doc_ids: dict[str, str] = {}
    docs: list[Document] = []
    items: list[QAItem] = []
    for row in ds:
        ctx = row["context"]
        if ctx not in doc_ids:
            doc_ids[ctx] = f"{row['title']}#{len(doc_ids)}"
            # The article title becomes a heading, so chunks carry it as context (see chunking.py).
            docs.append(Document(doc_ids[ctx], f"# {row['title'].replace('_', ' ')}\n\n{ctx}"))
        items.append(
            QAItem(
                row["question"],
                list(dict.fromkeys(row["answers"]["text"])),
                doc_ids[ctx],
            )
        )
    rng = random.Random(seed)
    answerable = [q for q in items if q.answerable]
    unanswerable = [q for q in items if not q.answerable]
    sample = rng.sample(answerable, n_per_type) + rng.sample(unanswerable, n_per_type)
    rng.shuffle(sample)
    return docs, sample


def bootstrap(values: list[float], n: int = 5000, seed: int = 0) -> list[float]:
    """95 % percentile-bootstrap interval of the mean."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return [float("nan"), float("nan")]
    idx = np.random.default_rng(seed).integers(0, x.size, (n, x.size))
    m = x[idx].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def evaluate_retrieval(pipe, items: list[QAItem], k_values=(1, 5, 10)) -> dict:
    """Retrieval metrics over *answerable* questions (unanswerable ones still have a gold paragraph,
    so we include them too; the paragraph is what a human would show)."""
    kmax = max(k_values)
    rows = {f"recall@{k}": [] for k in k_values} | {"mrr": [], "ndcg@10": []}
    for q in items:
        ranked = [h.chunk.doc_id for h in pipe.retriever.search(q.question, kmax)]
        for k in k_values:
            rows[f"recall@{k}"].append(recall_at_k(ranked, q.gold_doc, k))
        rows["mrr"].append(reciprocal_rank(ranked, q.gold_doc))
        rows["ndcg@10"].append(ndcg_at_k(ranked, q.gold_doc, 10))
    return {name: {"mean": float(np.mean(v)), "ci95": bootstrap(v)} for name, v in rows.items()}


def run_retrieval_ablation(args) -> dict:
    """BM25 vs dense (two embedding sizes) vs hybrid vs hybrid + reranker, on the same questions."""
    from ragservice.embeddings import build_embedder
    from ragservice.generation import ExtractiveLLM
    from ragservice.guardrails import looks_like_injection
    from ragservice.pipeline import RAGPipeline
    from ragservice.retrieval import CrossEncoderReranker

    docs, items = load_squad_v2(args.n, args.seed)
    results = {"n_questions": len(items), "n_docs": len(docs), "configs": {}}
    # Guardrail false positives: SQuAD passages are benign Wikipedia text, so every flagged one is a false alarm.
    flagged = sum(looks_like_injection(d.text) for d in docs)
    results["injection_screen_false_positive_rate"] = {"flagged": flagged, "passages": len(docs), "rate": flagged / max(len(docs), 1)}
    print("injection screen false positives:", json.dumps(results["injection_screen_false_positive_rate"]), flush=True)
    for emb_name in args.embedders:
        pipe = RAGPipeline(build_embedder(emb_name), ExtractiveLLM())
        pipe.ingest(docs)
        for mode in ("bm25", "dense", "hybrid"):
            if mode == "bm25" and emb_name != args.embedders[0]:
                continue  # BM25 doesn't depend on the embedder; run it once
            pipe.retriever.mode = mode
            label = "bm25" if mode == "bm25" else f"{mode} ({emb_name.split('/')[-1]})"
            results["configs"][label] = evaluate_retrieval(pipe, items)
            print(
                label,
                json.dumps({k: round(v["mean"], 3) for k, v in results["configs"][label].items()}),
                flush=True,
            )
        if args.reranker:
            pipe.retriever.mode = "hybrid"
            pipe.retriever.reranker = CrossEncoderReranker(args.reranker)
            label = f"hybrid ({emb_name.split('/')[-1]}) + {args.reranker.split('/')[-1]}"
            results["configs"][label] = evaluate_retrieval(pipe, items)
            print(
                label,
                json.dumps({k: round(v["mean"], 3) for k, v in results["configs"][label].items()}),
                flush=True,
            )
            pipe.retriever.reranker = None
    return results


def run_answer_eval(args) -> dict:
    """End-to-end answer quality with retrieval, versus the same model answering closed-book."""
    from ragservice.embeddings import build_embedder
    from ragservice.generation import ABSTAIN, HFLocalLLM
    from ragservice.pipeline import RAGPipeline

    docs, items = load_squad_v2(args.n, args.seed)
    llm = HFLocalLLM(args.generator)
    pipe = RAGPipeline(build_embedder(args.embedders[0]), llm, mode="hybrid", top_k=args.top_k)
    pipe.ingest(docs)

    # 1) Retrieve for every question, then generate in batches (much faster than one by one).
    retrieved = [pipe.retrieve(q.question)[0] for q in items]
    prompts = [build_prompt(q.question, hits) for q, hits in zip(items, retrieved)]
    outputs: list[str] = []
    for i in range(0, len(prompts), args.batch_size):
        outputs += llm.complete_batch(prompts[i : i + args.batch_size], max_tokens=128)

    # 2) Closed-book baseline: same model, same questions, no sources. Shows what retrieval adds.
    closed_prompts = [
        [
            {
                "role": "system",
                "content": f"Answer concisely. If you are not sure, reply exactly: {ABSTAIN}",
            },
            {"role": "user", "content": q.question},
        ]
        for q in items
    ]
    closed: list[str] = []
    for i in range(0, len(closed_prompts), args.batch_size):
        closed += llm.complete_batch(closed_prompts[i : i + args.batch_size], max_tokens=128)
    del llm, pipe.llm

    def score(answers: list[str], with_citations: bool) -> dict:
        abst = [is_abstention(a) for a in answers]
        ans_idx = [i for i, q in enumerate(items) if q.answerable]
        out = {
            "answerable_contains_ref": float(np.mean([answer_contains_reference(answers[i], items[i].answers) for i in ans_idx])),
            "answerable_f1": float(np.mean([f1(answers[i], items[i].answers) for i in ans_idx])),
            "answerable_em": float(np.mean([exact_match(answers[i], items[i].answers) for i in ans_idx])),
            **abstention_scores(abst, [q.answerable for q in items]),
        }
        out["answerable_contains_ref_ci95"] = bootstrap([answer_contains_reference(answers[i], items[i].answers) for i in ans_idx])
        if with_citations:
            non_abst = [i for i, a in enumerate(abst) if not a]
            out["answers_with_citation"] = (
                float(np.mean([bool(parse_citations(answers[i], len(retrieved[i]))) for i in non_abst])) if non_abst else 0.0
            )
            out["retrieval_recall@k"] = float(
                np.mean([recall_at_k([h.chunk.doc_id for h in r], q.gold_doc, args.top_k) for q, r in zip(items, retrieved)])
            )
        return out

    report = {
        "generator": args.generator,
        "embedder": args.embedders[0],
        "top_k": args.top_k,
        "n_questions": len(items),
        "rag": score(outputs, True),
        "closed_book": score(closed, False),
    }

    # 3) Faithfulness with an LLM judge, on answers that did not abstain.
    if args.judge:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
        judge = HFLocalLLM(args.judge)
        idx = [i for i, a in enumerate(outputs) if not is_abstention(a)]
        jprompts = []
        for i in idx:
            src = "\n\n".join(f"[{n}] {h.chunk.text}" for n, h in enumerate(retrieved[i], start=1))
            jprompts.append(
                [
                    {
                        "role": "user",
                        "content": FAITHFULNESS_PROMPT.format(sources=src, answer=outputs[i]),
                    }
                ]
            )
        verdicts: list[str] = []
        for j in range(0, len(jprompts), args.batch_size):
            verdicts += judge.complete_batch(jprompts[j : j + args.batch_size], max_tokens=4)
        scores = [parse_faithfulness(v) for v in verdicts]
        valid = [s for s in scores if s is not None]
        # Separate answerable vs unanswerable: unsupported answers to unanswerable questions are hallucinations.
        unans_valid = [s for s, i in zip(scores, idx) if s is not None and not items[i].answerable]
        report["judge"] = args.judge
        report["rag"]["faithfulness"] = float(np.mean(valid)) if valid else None
        report["rag"]["faithfulness_ci95"] = bootstrap(valid)
        report["rag"]["faithfulness_on_unanswerable_non_abstained"] = float(np.mean(unans_valid)) if unans_valid else None
        report["rag"]["judge_unparseable"] = sum(s is None for s in scores)

    samples = [
        {
            "question": q.question,
            "answerable": q.answerable,
            "references": q.answers,
            "rag_answer": a,
            "closed_book_answer": c,
        }
        for q, a, c in list(zip(items, outputs, closed))[:40]
    ]
    report["samples"] = samples
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=["retrieval", "answers"])
    parser.add_argument(
        "--n",
        type=int,
        default=300,
        help="questions per type (answerable / unanswerable)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--embedders",
        nargs="+",
        default=["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5"],
    )
    parser.add_argument("--reranker", default="BAAI/bge-reranker-base")
    parser.add_argument("--generator", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--judge", default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run_retrieval_ablation(args) if args.job == "retrieval" else run_answer_eval(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
