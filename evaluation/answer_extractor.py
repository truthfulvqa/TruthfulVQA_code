import re
from typing import Dict, Tuple

def extract_answer(response: str) -> str:
    """
    Extract the answer from the response.
    """
    # First try to find all (A) patterns and return the last one
    matches = re.findall(r'\(([A-Z])\)', response)
    if matches:
        return matches[-1]
    
    # Then try to find half-parentheses, like 'A)'
    matches = re.findall(r'([A-Z])\)', response)
    if matches:
        return matches[-1]
    
    # Then try to find the letter in boxed{}, like 'boxed{A}' or 'boxed{a}'
    matches = re.findall(r'boxed{([A-Za-z])}', response)
    if matches:
        return matches[-1].upper()
    
    # If no (A) patterns found, find all uppercase letters
    matches = re.findall(r'[A-Z]', response)
    if not matches:
        return ""
    
    # Get first and last uppercase letters
    first_upper = matches[0]
    last_upper = matches[-1]
    
    # Check if either letter is in ABCD
    valid_letters = ['A', 'B', 'C', 'D']
    first_valid = first_upper in valid_letters
    last_valid = last_upper in valid_letters
    
    # Return the appropriate letter based on the rules
    if first_valid and last_valid:
        return last_upper
    elif first_valid:
        return first_upper
    elif last_valid:
        return last_upper
    else:
        return last_upper

def extract_conclusion(response: str) -> str:
    """
    Extract the conclusion enclosed in <CONCLUSION> and </CONCLUSION>
    """
    conclusion_match = re.search(r'<CONCLUSION>(.*?)</CONCLUSION>', response, re.DOTALL)
    if conclusion_match:
        return conclusion_match.group(1).strip()
    else:
        return ""

def extract_reasoning_and_answer(response: str) -> Tuple[str, str]:
    """
    Extract reasoning and answer from the response.
    Returns a tuple of (reasoning, answer)
    """
    reasoning = ""
    answer = ""
    
    # Extract reasoning between ◁think▷ and ◁/think▷
    reasoning_match = re.search(r'◁think▷(.*?)◁/think▷', response, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Extract answer after ◁/think▷
    answer_match = re.search(r'◁/think▷(.*?)$', response, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    
    return reasoning, answer

def process_responses(input_file: str, output_file: str):
    """
    Process responses from input file and save results to output file.
    """
    import json
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for item in data:
        response = item["response"]
        extracted_answer = extract_answer(response)
        results.append({
            "category": item["category"],
            "subcategory": item["subcategory"],
            "level": item["level"],
            "ground_truth": item["ground_truth"],
            "extracted_answer": extracted_answer,
            "is_correct": item["ground_truth"] == extracted_answer,
            "response": response,
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4) 