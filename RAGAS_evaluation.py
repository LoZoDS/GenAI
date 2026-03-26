import re
import numpy as np

# === PART 1: Load rag_retrieval_chain.py as module to re-use functions and model ===#

print("Importing rag_retrieval_chain...")
import rag_retrieval_chain as rag_module

retriever        = rag_module.retriever
rag_chain        = rag_module.rag_chain
llm              = rag_module.llm              # reuse loaded Qwen — no double load
ask_chatbot      = rag_module.ask_chatbot
retrieve_context = rag_module.retrieve_context

print("✅ rag_module module loaded\n")


# ==== PART 2: Create LLM Judge ===#

def llm_judge(prompt: str) -> str:
    """With this function the program runs a prompt through the Qwen model
    and returns string response."""
    return llm.invoke(prompt)


def extract_yes_no(text: str) -> bool:
    """Return True if the response contains YES, False if NO (or unclear)."""
    t = text.strip().upper()
    # Check the very first word first (most reliable for small models)
    first_word = t.split()[0] if t.split() else ""
    if first_word in ("YES", "NO"):
        return first_word == "YES"
    # Fall back to searching anywhere in the text
    has_yes = "YES" in t
    has_no  = "NO"  in t
    if has_yes and not has_no:
        return True
    return False  # default to NO when uncertain


def extract_score(text: str, fallback: float = 0.5) -> float:
    """Pull a 0–1 float out of Qwen's response."""
    # Look for decimal like 0.8 or 0.75
    match = re.search(r'\b(0\.\d+|1\.0|1)\b', text)
    if match:
        val = float(match.group(1))
        return min(max(val, 0.0), 1.0)
    # Look for integer score out of 10 (e.g. "8/10" or "8 out of 10")
    match = re.search(r'\b([0-9]|10)\s*(?:/|out of)\s*10\b', text.lower())
    if match:
        return int(match.group(1)) / 10.0
    return fallback


#=== PART 3: Define evaluation questions and their answers ===#

eval_questions = [
    {
        "question":     "How much milk can a 2-month-old drink at each meal?",
        "ground_truth": "Babies can drink up to 180 ml at each meal, six times a day.",
    },
    {
        "question":     "What cooing behaviour is expected at 2 months?",
        "ground_truth": "A 2-month-old is making cooing noises and parents can encourage them by speaking back.",
    },
    {
        "question":     "How can parents support tummy time for a 2-month-old?",
        "ground_truth": "Parents should face the baby and speak with them while on their stomach, and move safe toys to encourage movement.",
    },
    {
        "question":     "What physical milestone can a 2-month-old achieve on their belly?",
        "ground_truth": "A 2-month-old can push up while on their belly and hold their head up.",
    },
    {
        "question":     "What social behaviour does a 2-month-old show?",
        "ground_truth": "A 2-month-old starts to smile at others, looks at parents, and pays attention to faces.",
    },
]


# === PART 4: Define RAGAS metrics ===#

# ── Metric 1: Faithfulness ───────────────────────────────────────────────────
def compute_faithfulness(answer: str, contexts: list[str]) -> dict:
    """
    Are all claims in the answer supported by the retrieved context?
    Score = supported_claims / total_claims

    Step 1: Ask Qwen to list every factual claim in the answer.
    Step 2: For each claim ask Qwen YES/NO — is it in the context?
    Step 3: supported / total = score.
    """
    clean = answer.split("\n\nNote:")[0].strip()

    context_block = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))

    # Step 1: extract claims
    claims_prompt = (
        f"List every factual claim in this answer. "
        f"Write each claim on a new line starting with '-'.\n\n"
        f"Answer: {clean}"
    )
    claims_raw = llm_judge(claims_prompt)
    claims = [
        line.lstrip("-• ").strip()
        for line in claims_raw.split("\n")
        if line.strip().startswith(("-", "•")) and len(line.strip()) > 5
    ]
    if not claims:
        claims = [clean]

    # Step 2: verify each claim
    supported = 0
    verdicts  = []
    for claim in claims:
        verify_prompt = (
            f"Context:\n{context_block}\n\n"
            f"Is this claim supported by the context above?\n"
            f"Claim: {claim}\n\n"
            f"Reply with YES or NO only."
        )
        response = llm_judge(verify_prompt)
        ok = extract_yes_no(response)
        verdicts.append({"claim": claim, "supported": ok})
        if ok:
            supported += 1

    score = supported / len(claims) if claims else 0.0
    return {
        "score":            score,
        "total_claims":     len(claims),
        "supported_claims": supported,
        "verdicts":         verdicts,
    }


# ── Metric 2: Answer Relevancy ───────────────────────────────────────────────
def compute_answer_relevancy(question: str, answer: str) -> dict:
    """
    Does the answer address the question?
    Ask Qwen to score 0.0–1.0. Disclaimers are expected and should be ignored.
    """
    # Strip disclaimer so it doesn't confuse the small model
    clean = answer.split("\n\nNote:")[0].strip()

    prompt = (
        f"Question: {question}\n\n"
        f"Answer: {clean}\n\n"
        f"How well does the answer address the question?\n"
        f"Give a score from 0.0 (not relevant) to 1.0 (fully relevant).\n"
        f"Reply with only a number like: 0.8"
    )
    response = llm_judge(prompt)
    score    = extract_score(response)
    return {"score": score, "raw_response": response}


# ── Metric 3: Context Precision ──────────────────────────────────────────────
def compute_context_precision(question: str, contexts: list[str],
                               ground_truth: str) -> dict:
    """
    Were the retrieved chunks useful, and were the best ones ranked first?
    Rank-weighted: relevant chunk at rank 1 scores more than at rank 3.
    """
    verdicts = []
    for i, ctx in enumerate(contexts):
        prompt = (
            f"Question: {question}\n"
            f"Correct answer: {ground_truth}\n\n"
            f"Context: {ctx}\n\n"
            f"Does this context help answer the question?\n"
            f"Reply with YES or NO only."
        )
        response   = llm_judge(prompt)
        is_relevant = extract_yes_no(response)
        verdicts.append({"rank": i + 1, "relevant": is_relevant})

    running  = 0
    prec_sum = 0.0
    for v in verdicts:
        if v["relevant"]:
            running  += 1
            prec_sum += running / v["rank"]

    total_relevant = sum(1 for v in verdicts if v["relevant"])
    score = prec_sum / max(total_relevant, 1)

    return {"score": score, "verdicts": verdicts}


# ── Metric 4: Context Recall ─────────────────────────────────────────────────
def compute_context_recall(contexts: list[str], ground_truth: str) -> dict:
    """
    Did we retrieve all the information needed to answer correctly?
    Split ground truth into sentences, check each one against the contexts.
    Score = attributed_sentences / total_sentences
    """
    context_block = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    sentences     = [s.strip() for s in re.split(r'[.!?]+', ground_truth)
                     if s.strip()]

    if not sentences:
        return {"score": 1.0, "total_sentences": 0, "attributed": 0, "sentences": []}

    attributed = 0
    verdicts   = []
    for sent in sentences:
        prompt = (
            f"Contexts:\n{context_block}\n\n"
            f"Is this statement found in the contexts above?\n"
            f"Statement: {sent}\n\n"
            f"Reply with YES or NO only."
        )
        response = llm_judge(prompt)
        ok = extract_yes_no(response)
        verdicts.append({"sentence": sent, "attributed": ok})
        if ok:
            attributed += 1

    score = attributed / len(sentences)
    return {
        "score":           score,
        "total_sentences": len(sentences),
        "attributed":      attributed,
        "sentences":       verdicts,
    }


# === PART 5: Evaluation Loop ===#

def run_rag(query: str) -> dict:
    """Run ask_chatbot and also capture the raw retrieved chunks."""
    docs, _  = retrieve_context(query, retriever)
    contexts = [doc.page_content for doc in docs]
    answer   = ask_chatbot(query, retriever, rag_chain)
    return {"question": query, "contexts": contexts, "answer": answer}


def run_evaluation():
    metrics = ["faithfulness", "answer_relevancy",
               "context_precision", "context_recall"]

    print("=" * 65)
    print("  RAGAS Evaluation")
    print("  Source : UNICEF Developmental Milestones")
    print("  Chain  : rag_retrieval_chain.py")
    print("  Judge  : Qwen/Qwen2.5-3B-Instruct (reused)")
    print("=" * 65, "\n")

    results = []

    for i, item in enumerate(eval_questions):
        question     = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{i+1}/{len(eval_questions)}] {question}")

        # Run RAG
        print("    → RAG pipeline...",      end=" ", flush=True)
        sample = run_rag(question)
        sample["ground_truth"] = ground_truth
        print(f"done  ({len(sample['contexts'])} chunks retrieved)")
        print(f"       Answer: {sample['answer'][:80].strip()}...")

        # Score metrics
        print("    → Faithfulness...",      end=" ", flush=True)
        faith  = compute_faithfulness(sample["answer"], sample["contexts"])
        print(f"{faith['score']:.2f}  ({faith['supported_claims']}/{faith['total_claims']} claims)")

        print("    → Answer Relevancy...",  end=" ", flush=True)
        relev  = compute_answer_relevancy(sample["question"], sample["answer"])
        print(f"{relev['score']:.2f}")

        print("    → Context Precision...", end=" ", flush=True)
        c_prec = compute_context_precision(sample["question"], sample["contexts"], ground_truth)
        print(f"{c_prec['score']:.2f}")

        print("    → Context Recall...",    end=" ", flush=True)
        c_rec  = compute_context_recall(sample["contexts"], ground_truth)
        print(f"{c_rec['score']:.2f}  ({c_rec['attributed']}/{c_rec['total_sentences']} sentences)")

        results.append({
            "question":          question,
            "faithfulness":      faith["score"],
            "answer_relevancy":  relev["score"],
            "context_precision": c_prec["score"],
            "context_recall":    c_rec["score"],
        })
        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    labels = ["Faithfulness", "Ans.Relevancy", "Ctx.Precision", "Ctx.Recall"]
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    header = f"{'':6}" + "".join(f"{l:>15}" for l in labels)
    print(header)
    print("-" * len(header))
    for i, row in enumerate(results):
        print(f"  Q{i+1}   " + "".join(f"{row[m]:>15.2f}" for m in metrics))
    print("-" * len(header))
    means = {m: np.mean([r[m] for r in results]) for m in metrics}
    print(f"  AVG  " + "".join(f"{means[m]:>15.2f}" for m in metrics))
    print()

    # ── Overall score (harmonic mean) ──────────────────────────────────────────
    overall = len(metrics) / sum(1.0 / max(means[m], 0.001) for m in metrics)
    print(f"  Overall RAGAS score (harmonic mean): {overall:.2f}")
    print()

    # ── Health check ───────────────────────────────────────────────────────────
    retriever_score = (means["context_precision"] + means["context_recall"]) / 2
    generator_score = (means["faithfulness"] + means["answer_relevancy"]) / 2

    print("  Component health:")
    print(f"    Retriever : {retriever_score:.2f}  {'✅' if retriever_score >= 0.7 else '⚠️ '}")
    print(f"    Generator : {generator_score:.2f}  {'✅' if generator_score >= 0.7 else '⚠️ '}")
    print()

    if retriever_score < 0.7:
        print("  → Retriever: try increasing k or improving chunking.")
    if generator_score < 0.7:
        print("  → Generator: tighten the prompt or try a larger model.")
    if means["context_recall"] < 0.6:
        print("  → Recall low: some facts aren't being retrieved — increase k.")
    if means["faithfulness"] < 0.6:
        print("  → Faithfulness low: Qwen may be adding info beyond the context.")

    print()
    print("=" * 65)
    print("  Done.")
    print("=" * 65)

    return results

if __name__ == "__main__":
    run_evaluation()