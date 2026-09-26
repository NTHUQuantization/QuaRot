"""Frozen conservative scorers; call score_math_final on FINAL text only.
Legacy extraction/normalization copied verbatim; no import of the GPU runtime.
"""
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/[+-]?\d[\d,]*)?")

def last_boxed(text: str) -> str | None:
    starts = [match.end() for match in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None
    start = starts[-1]
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return None

def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    return text.replace("<think>", "").replace("</think>", "")

def extract_numeric_answer(text: str) -> str | None:
    text = strip_thinking(text)
    boxed = last_boxed(text)
    if boxed is not None:
        values = NUMBER_RE.findall(boxed)
        if values:
            return values[-1]
    marker = re.findall(
        r"(?:final\s+answer|answer\s+is|####)\s*[:=]?\s*([^\n]+)",
        text, flags=re.I)
    if marker:
        values = NUMBER_RE.findall(marker[-1])
        if values:
            return values[-1]
    values = NUMBER_RE.findall(text)
    return values[-1] if values else None

def numeric_value(value: str | None) -> Fraction | None:
    if value is None:
        return None
    value = value.replace(",", "").strip()
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        return Fraction(Decimal(value))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None

def normalize_math(value: str | None) -> str | None:
    if value is None:
        return None
    value = strip_thinking(value).strip().strip("$.")
    boxed = last_boxed(value)
    if boxed is not None:
        value = boxed
    value = re.sub(r"\\(?:left|right)", "", value)
    value = re.sub(r"\\(?:,|!|;|quad|qquad)", "", value)
    value = value.replace(" ", "").replace("\n", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace("^{\\circ}", "^\\circ").replace("^\u00b0", "^\\circ")
    return value.lower()

def math_equal(predicted: str | None, expected: str) -> bool:
    left, right = normalize_math(predicted), normalize_math(expected)
    if left is None:
        return False
    if left == right:
        return True
    left_number = numeric_value(extract_numeric_answer(left))
    right_number = numeric_value(extract_numeric_answer(right))
    return (left_number is not None and right_number is not None
            and left_number == right_number
            and NUMBER_RE.fullmatch(left) is not None
            and NUMBER_RE.fullmatch(right) is not None)

def python_prompt(prompt: str) -> str:
    match = re.search(r"(?m)^(?:from\s+\S+\s+import|import\s+|def\s+|class\s+)", prompt)
    if not match:
        raise ValueError("HumanEval prompt has no Python source start")
    return prompt[match.start():]

def code_candidate(prompt: str, completion: str, name: str) -> str:
    completion = strip_thinking(completion)
    # A token limit can cut off the closing fence. Preserve imports above the
    # function in that final block instead of falling back to the `def` line.
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)(?:```|(?m:^[ \t]*`{1,2}[ \t]*\Z)|\Z)", completion,
                        flags=re.S | re.I)
    preferred = [block for block in blocks
                 if re.search(rf"(?m)^def\s+{re.escape(name)}\s*\(", block)]
    if preferred:
        return preferred[-1].strip() + "\n"
    if blocks:
        block = max(blocks, key=len).strip()
        if re.search(r"(?m)^def\s+", block):
            return block + "\n"
    function = re.search(
        rf"(?ms)^def\s+{re.escape(name)}\s*\(.*", completion)
    if function:
        return function.group(0).strip() + "\n"
    return python_prompt(prompt).rstrip() + "\n" + completion.rstrip() + "\n"


def boxed_numeric_value(text):
    if text is None:
        return None
    text = text.strip().strip('$').strip().replace(' ', '')
    text = text.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    frac = re.fullmatch(r'([+-]?)\\frac\{([+-]?[\d,.]+)\}\{([+-]?[\d,.]+)\}', text)
    if frac:
        sign, numerator, denominator = frac.groups()
        value = numeric_value(numerator + '/' + denominator)
        return -value if sign == '-' and value is not None else value
    if NUMBER_RE.fullmatch(text) is None:
        return None
    return numeric_value(text)


def score_math_final(task, final_text, expected):
    """Caller must already reject incomplete thinking using common.final_ids."""
    predicted = last_boxed(final_text)
    if task == 'gsm8k':
        left, right = boxed_numeric_value(predicted), boxed_numeric_value(expected)
        if right is None:
            raise ValueError('Invalid ground truth numeric answer')
        correct = left is not None and left == right
        extraction_failure = left is None
    elif task == 'math_500':
        correct = predicted is not None and math_equal(predicted, expected)
        extraction_failure = predicted is None or not predicted.strip()
    else:
        raise ValueError(task)
    return {'prediction': predicted, 'correct': bool(correct),
            'extraction_failure': extraction_failure}
