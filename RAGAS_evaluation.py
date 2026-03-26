import re
import numpy as np

# === PART 1: Load module ===#

print("Importing rag_retrieval_chain...")
# Here we import the module where we already defined our RAG pipeline.
# This allows us to reuse everything instead of rewriting it.
import rag_retrieval_chain as rag_module
retriever        = rag_module.retriever                 #retrieves relevant chunks from the knowledge base
rag_chain        = rag_module.rag_chain                 #full pipeline (retrieval + generation)
llm              = rag_module.llm                       #language model (latest Microsoft)
ask_chatbot      = rag_module.ask_chatbot               #function to run full RAG pipeline
retrieve_context = rag_module.retrieve_context          #function to retrieve context
print("✅ rag_module module loaded\n")


# ==== PART 2: Create LLM Judge ===#

def llm_judge(prompt: str) -> str:
    """Function sends a prompt to the judge llm model and returns the text
    the model generates"""
    return llm.invoke(prompt)


def extract_yes_no(text: str) -> bool:
    """This function tries to interpret the model's answer as YES or NO.
    It defaults to NO if answer is unclear"""
    t = text.strip().upper() # clean and standarize the text
    # Check if the first word is "YES" or "NO"
    first_word = t.split()[0] if t.split() else ""
    if first_word in ("YES", "NO"):
        return first_word == "YES"
    # Fall back to searching anywhere in the text
    has_yes = "YES" in t
    has_no  = "NO"  in t
    if has_yes and not has_no:
        return True
    return False  # default to NO when uncertain


def extract_score(text: str, fallback: float = 0.0) -> float:
    """This function extracts a score between 0 and 1 from the model's response.
    If no valid score is found, it returns a fallback value of 0"""
    # Look for decimal like 0.8 or 0.75 score in the response from Qwen
    match = re.search(r'\b(0\.\d+|1\.0|1)\b', text)
    if match:
        val = float(match.group(1))
        return min(max(val, 0.0), 1.0)
    # Look for integer like 8 or 7 score in the response from Qwen
    match = re.search(r'\b([0-9]|10)\s*(?:/|out of)\s*10\b', text.lower())
    if match:
        return int(match.group(1)) / 10.0
    return fallback


#=== PART 3: Define evaluation questions and their answers ===#
#These are the questions and answers we chose from our knowledge base to compare the answer
#from our RAG pipeline with the "true" answer we got from our sources.
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
# Metric 1: Faithfulness
def compute_faithfulness(answer: str, contexts: list[str]) -> dict:
    """
    This function gets a faithfulness score by [1] asking the judge to list all factual claims
    in the answer, [2] for each claim check if it is in the context, [3] calculating
    supported_claims / total_claims
    """
    clean = answer.split("\n\nNote:")[0].strip() # remove disclaimer part if it exists

    # This builds a context block that is easy to read.
    context_block = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))

    # Step 1: ask the model to list all factual claims
    claims_prompt = (
        # after researching we decided it was better to create a direct simple prompt for a judge
        # that uses a very simple model like Qwen.
        f"List every factual claim in this answer. " 
        f"Write each claim on a new line starting with '-'.\n\n"
        f"Answer: {clean}"
    )
    claims_raw = llm_judge(claims_prompt)
    claims = [ # clean the claims. If already clean, then keep it as it is.
        line.lstrip("-• ").strip()
        for line in claims_raw.split("\n")
        if line.strip().startswith(("-", "•")) and len(line.strip()) > 5
    ]
    if not claims:
        claims = [clean]

    # Step 2: check each claim
    supported = 0
    verdicts  = []
    for claim in claims:
        verify_prompt = ( # prompt to pass the judge to verify the claim
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


# Metric 2: Answer Relevancy
def compute_answer_relevancy(question: str, answer: str) -> dict:
    """
    This check tells how well the answer actually addresses the question.
    """
    # Strip disclaimer so it doesn't confuse the small model
    clean = answer.split("\n\nNote:")[0].strip()

    # Ask the model how relevant the answer for the question is.
    # Response should be a number like 0.8
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


# Metric 3: Context Precision
def compute_context_precision(question: str, contexts: list[str],
                               ground_truth: str) -> dict:
    """
    This check tells us whether the retrieved chunks are useful to answer the question,
    and whether the most useful ones appear earlier.
    """
    verdicts = [] # store whether each context chunk is relevant or not

    # This loops through the retrieved context chunks
    for i, ctx in enumerate(contexts):

        # Ask the judge model if the chunk helps answer the question
        prompt = (
            f"Question: {question}\n"
            f"Correct answer: {ground_truth}\n\n"
            f"Context: {ctx}\n\n"
            f"Does this context help answer the question?\n"
            f"Reply with YES or NO only."
        )
        response   = llm_judge(prompt) # store the answer of the judge
        is_relevant = extract_yes_no(response) # convert the answer into True (YES) or False (NO)
        # The line below saves the result of the chunk with the rank
        verdicts.append({"rank": i + 1, "relevant": is_relevant})

    running  = 0 # to track the number of relevant chunks
    precision_sum = 0.0 # to accumulate the precision values

    for v in verdicts:
        # This considers only the chunks that are relevant
        if v["relevant"]:
            running  += 1
            precision_sum += running / v["rank"] # to calculate the number of relevant chunks so far /  current position in list

    total_relevant = sum(1 for v in verdicts if v["relevant"]) # total number of relevant
    # Final score = average precision across all relevant chunks
    # We use max(total_relevant, 1) to avoid division by zero if no chunks are relevant
    score = precision_sum / max(total_relevant, 1)

    return {"score": score, "verdicts": verdicts}


# ── Metric 4: Context Recall ─────────────────────────────────────────────────
def compute_context_recall(contexts: list[str], ground_truth: str) -> dict:
    """
    This checks whether all important information from the ground truth
    is present in the retrieved context.
    """
    context_block = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    sentences     = [s.strip() for s in re.split(r'[.!?]+', ground_truth)
                     if s.strip()]

    if not sentences:
        return {"score": 1.0, "total_sentences": 0, "attributed": 0, "sentences": []}

    attributed = 0 # count how many sentences are found in the context
    verdicts   = [] # store results for each sentence
    for sent in sentences:
        # Ask the LLM if this sentence is present in the contexts
        prompt = (
            f"Contexts:\n{context_block}\n\n"
            f"Is this statement found in the contexts above?\n"
            f"Statement: {sent}\n\n"
            f"Reply with YES or NO only."
        )
        response = llm_judge(prompt) # actual send to llm
        ok = extract_yes_no(response) # convert response to True/False
        verdicts.append({"sentence": sent, "attributed": ok})
        if ok:
            attributed += 1

    # Final score = proportion of ground truth sentences covered by context
    score = attributed / len(sentences)
    return {
        "score":           score,
        "total_sentences": len(sentences),
        "attributed":      attributed,
        "sentences":       verdicts,
    }


# === PART 5: Evaluation Loop ===#

def run_rag(query: str) -> dict:
    """This function runs the RAG pipeline and also returns the retrieved contexts."""
    docs, _  = retrieve_context(query, retriever)
    contexts = [doc.page_content for doc in docs]
    answer   = ask_chatbot(query, retriever, rag_chain)
    return {"question": query, "contexts": contexts, "answer": answer}


def run_evaluation():
    """
    This function runs the full evaluation over all questions and prints detailed results.
    """
    metrics = ["faithfulness", "answer_relevancy",
               "context_precision", "context_recall"]

    print("=" * 65)
    print("  RAGAS Evaluation")
    print(f"  Judge  : {llm} (reused)")
    print("=" * 65, "\n")

    results = []

    for i, item in enumerate(eval_questions):
        question     = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{i+1}/{len(eval_questions)}] {question}")

        # Run RAG
        print("Running RAG pipeline",      end=" ", flush=True)
        sample = run_rag(question)
        sample["ground_truth"] = ground_truth
        print(f"done  ({len(sample['contexts'])} chunks retrieved)")
        print(f"       Answer: {sample['answer'][:80].strip()}...")

        # Score metrics
        print("Checking Faithfulness...",      end=" ", flush=True)
        faith  = compute_faithfulness(sample["answer"], sample["contexts"])
        print(f"{faith['score']:.2f}  ({faith['supported_claims']}/{faith['total_claims']} claims)")

        print("Checking Answer Relevancy...",  end=" ", flush=True)
        relev  = compute_answer_relevancy(sample["question"], sample["answer"])
        print(f"{relev['score']:.2f}")

        print("Checking Context Precision...", end=" ", flush=True)
        c_prec = compute_context_precision(sample["question"], sample["contexts"], ground_truth)
        print(f"{c_prec['score']:.2f}")

        print("Checking Context Recall...",    end=" ", flush=True)
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