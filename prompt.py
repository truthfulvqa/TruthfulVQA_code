MULTIPLE_CHOICE_PROMPT = """
<image>\n{question}\n\n{options}\n\nAnswer with the option's letter enclosed in () at the end of your response.
"""

MULTIPLE_CHOICE_ECE_PROMPT = """
<image>\n{question}\n\n{options}\n\nAnswer with the option's letter enclosed in () at the end of your response. Give your confidence score of your answer (a fractional number in the range of 0-1) enclosed in [] at the end of your response.\n\n Example Output (A)[0.9]
"""

OPEN_ENDED_PROMPT = """
<image>\n{question}
"""

def format_options(options: list[str]) -> str:
    formatted_options = []
    for i, option in enumerate(options):
        formatted_options.append(f"({chr(65 + i)}) {option}")
    return "\n".join(formatted_options)

def format_multiple_choice_prompt(question: str, options: list[str]) -> str:
    return MULTIPLE_CHOICE_PROMPT.format(question=question, options=format_options(options))

def format_multiple_choice_ece_prompt(question: str, options: list[str]) -> str:
    return MULTIPLE_CHOICE_ECE_PROMPT.format(question=question, options=format_options(options))

def format_open_ended_prompt(question: str) -> str:
    return OPEN_ENDED_PROMPT.format(question=question)