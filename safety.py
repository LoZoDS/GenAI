import re
from prompts import FALLBACKS, DISCLAIMER

URGENT_PATTERNS = [
    r"not breathing",
    r"seizure",
    r"unconscious",
    r"passed out",
    r"severe injury",
    r"emergency",
]

DIAGNOSTIC_PATTERNS = [
    r"\bautism\b",
    r"\badhd\b",
    r"\bdisorder\b",
    r"\bdiagnose\b",
    r"\bdiagnosis\b",
    r"\bdelay\b",
    r"\bis this normal\b",
    r"\bwhat is wrong with my child\b",
    r"\bdoes my child have\b",
    r"\bmy child has\b",
]

OUT_OF_SCOPE_PATTERNS = [
    r"\bmedication\b",
    r"\bmedicine\b",
    r"\bdose\b",
    r"\btherapy\b",
    r"\blegal advice\b",
    r"\bfinancial advice\b",
]

UNSAFE_ANSWER_PATTERNS = [
    r"\byour child has\b",
    r"\bthis means your child has\b",
    r"\bthe diagnosis is\b",
    r"\bi diagnose\b",
    r"\bdefinitely has\b",
    r"\bcertainly has\b",
    r"\bprescribe\b",
    r"\badminister\b",
    r"\byou should give medication\b",
]

def classify_question(question: str) -> str:
    q = question.lower()

    for pattern in URGENT_PATTERNS:
        if re.search(pattern, q):
            return "urgent"

    for pattern in DIAGNOSTIC_PATTERNS:
        if re.search(pattern, q):
            return "diagnostic"

    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, q):
            return "out_of_scope"

    return "in_scope"

def clean_model_output(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    if "Assistant:" in text:
        text = text.rsplit("Assistant:", 1)[-1].strip()

    if text.startswith("Human:"):
        text = text.replace("Human:", "", 1).strip()

    return text

def remove_trailing_incomplete_list_item(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\n?\s*\d+\.\s*$", "", text)
    return text.strip()

def validate_answer(answer: str) -> str:
    if not answer or not answer.strip():
        return FALLBACKS["no_evidence"]

    answer = clean_model_output(answer)
    answer = remove_trailing_incomplete_list_item(answer)

    lower_answer = answer.lower()

    for pattern in UNSAFE_ANSWER_PATTERNS:
        if re.search(pattern, lower_answer):
            return FALLBACKS["diagnostic"]

    if DISCLAIMER not in answer:
        answer = answer.strip() + "\n\n" + DISCLAIMER

    return answer