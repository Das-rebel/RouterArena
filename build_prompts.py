#!/usr/bin/env python3
"""Generate properly formatted RouterArena dataset JSON files."""
import sys, os, json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) 

from datasets import load_dataset, load_from_disk

def escape_format_braces(text):
    if not isinstance(text, str):
        return text
    result = ""
    i = 0
    while i < len(text):
        if text[i] == "{":
            if i + 1 < len(text) and text[i + 1] == "{":
                result += "{{"
                i += 2
            else:
                result += "{{"
                i += 1
        elif text[i] == "}":
            if i + 1 < len(text) and text[i + 1] == "}":
                result += "}}"
                i += 2
            else:
                result += "}}"
                i += 1
        else:
            result += text[i]
            i += 1
    return result

def safe_format_prompt(prompt_template, **kwargs):
    escaped_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            escaped_kwargs[key] = escape_format_braces(value)
        else:
            escaped_kwargs[key] = value
    return prompt_template.format(**escaped_kwargs)

# Default templates (same as in prep_datasets.py)
DEFAULT_MCQ_TEMPLATE = """Please read the following multiple-choice questions and provide the most likely correct answer based on your knowledge.

Context: {Context}

{Question}

{Options}


Please format your final answer as the single letter of the correct option (A, B, C, D, E, F, G, H, I, or J) in the last line."""

DEFAULT_OPEN_TEMPLATE = """Please answer the following question based on your knowledge.

Context: {Context}

{Question}"""

def format_options(options_list):
    if not options_list:
        return ""
    labels = [chr(65 + i) for i in range(len(options_list))]
    return "\n".join(f"{label}. {opt}" for label, opt in zip(labels, options_list))

def build_formatted_prompts(split_name):
    """Build formatted prompts from RouterArena dataset."""
    ds = load_dataset("RouteWorks/RouterArena", split=split_name)
    
    formatted = []
    for row in ds:
        question = row.get("Question", "")
        options = row.get("Options", [])
        context = row.get("Context", "")
        global_index = row.get("Global Index")
        
        context_str = str(context) if context and str(context).strip() else "None"
        
        if options and len(options) > 0:
            options_str = format_options(options)
            prompt = safe_format_prompt(
                DEFAULT_MCQ_TEMPLATE,
                Context=context_str,
                Question=question,
                Options=options_str,
            )
        else:
            prompt = safe_format_prompt(
                DEFAULT_OPEN_TEMPLATE,
                Context=context_str,
                Question=question,
            )
        
        formatted.append({
            "prompt_formatted": prompt,
            "global index": global_index,
        })
    
    return formatted

# Build and save
out_dir = "./dataset"
os.makedirs(out_dir, exist_ok=True)

for split in ["sub_10", "full"]:
    print(f"Building {split} prompts...")
    data = build_formatted_prompts(split)
    filename = "router_data_10.json" if split == "sub_10" else "router_data.json"
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(data)} items to {path}")
    # Show first prompt
    print(f"  Sample: {data[0]['prompt_formatted'][:150]}...")
    print()
