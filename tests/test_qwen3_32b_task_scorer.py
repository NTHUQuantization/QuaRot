from e2e.score_qwen3_32b_tasks import (
    code_candidate,
    extract_math_answer,
    extract_numeric_answer,
    last_boxed,
    math_equal,
    numeric_value,
    python_prompt,
)


def test_last_boxed_preserves_nested_latex():
    text = r"work \boxed{1} then \boxed{\left(3,\frac{\pi}{2}\right)}"
    assert last_boxed(text) == r"\left(3,\frac{\pi}{2}\right)"
    assert extract_math_answer(text) == r"\left(3,\frac{\pi}{2}\right)"


def test_numeric_answer_prefers_final_marker_and_normalizes():
    text = "first 10 then final answer: 1,234.50"
    assert extract_numeric_answer(text) == "1,234.50"
    assert numeric_value("1,234.50") == numeric_value("1234.5")
    assert numeric_value("3/4") == numeric_value("0.75")


def test_math_equal_is_conservative():
    assert math_equal(r"\boxed{\left( 3, \frac{\pi}{2} \right)}",
                      r"\left(3,\frac{\pi}{2}\right)")
    assert math_equal("0.75", "3/4")
    assert not math_equal(r"\frac{2}{4}", r"\frac{1}{2}")


def test_code_candidate_prefers_named_python_fence():
    prompt = "Complete the code I provided.\n\ndef f(x):\n    \"\"\"doc\"\"\"\n"
    completion = "Explanation.\n```python\ndef f(x):\n    return x + 1\n```"
    assert code_candidate(prompt, completion, "f") == (
        "def f(x):\n    return x + 1\n")
    assert python_prompt(prompt).startswith("def f")


def test_code_candidate_appends_body_continuation():
    prompt = "Instruction\n\ndef f(x):\n    \"\"\"doc\"\"\"\n"
    result = code_candidate(prompt, "    return x * 2", "f")
    assert result.startswith("def f")
    assert result.endswith("    return x * 2\n")
