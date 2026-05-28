#!/usr/bin/env python3
"""Build properly formatted RouterArena prompts using official config templates."""
import sys, os, json
from typing import Dict, Any, List

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

def build_formatted_prompts(split_name: str) -> List[Dict[str, Any]]:
    ds = load_dataset("RouteWorks/RouterArena")[split_name]
    
    # Load all dataset configs (same as prep_datasets.py)
    config_dir = "./config/eval_config/zero-shot"
    dataset_names = [
        "AIME","ArcMMLU","AsDiv","ChessInstruct_mcq","ChessInstruct",
        "Ethics_commonsense","Ethics_deontology","Ethics_justice","Ethics_virtue",
        "FinQA","GeoBench","GeoGraphyData","GSM8K","LiveCodeBench","MATH","MathQA",
        "MedMCQA","MMLUPro","MMLU","MusicTheoryBench","NarrativeQA","OpenTDB",
        "PubMedQA","QANTA","SocialiQA","SuperGLUE-CausalReasoning","SuperGLUE-ClozeTest",
        "SuperGLUE-Entailment","SuperGLUE-QA","SuperGLUE-RC","SuperGLUE-Wic","SuperGLUE-Wsc",
        "WMT19-cs-en","WMT19-de-en","WMT19-fi-en","WMT19-gu-en","WMT19-kk-en",
        "WMT19-lt-en","WMT19-ru-en","WMT19-zh-en",
    ]
    dataset_configs = {}
    for dn in dataset_names:
        cfg_path = os.path.join(config_dir, f"{dn}.json")
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
                dataset_configs[dn] = cfg.get("eval_params", {})
        except FileNotFoundError:
            continue
    
    formatted = []
    for row in ds:
        global_index = row.get("Global Index") or row.get("global_index") or row.get("global index")
        question = row.get("Question", "")
        context = row.get("Context", "")
        options = row.get("Options", [])
        
        # Determine dataset name from Global Index
        parts = global_index.split("_") if global_index else [""]
        if parts[0] == "Ethics" and len(parts) >= 2:
            base_name = f"{parts[0]}_{parts[1]}"
        elif parts[0] == "ChessInstruct" and len(parts) >= 2:
            if options and len(options) > 0:
                base_name = "ChessInstruct_mcq"
            else:
                base_name = "ChessInstruct"
        elif row.get("Dataset name"):
            full = row["Dataset name"]
            base_name = str(full).split("_", 1)[0] if "_" in str(full) else str(full)
        else:
            base_name = parts[0]
        
        # Get eval params
        eval_params = dataset_configs.get(base_name, {})
        prompt_template = eval_params.get("prompt", "{Question}")
        
        # Build options string
        if options and len(options) > 0:
            labels = [chr(65 + i) for i in range(len(options))]
            options_str = "\n".join(f"{l}. {o}" for l, o in zip(labels, options))
        else:
            options_str = ""
        
        context_str = str(context) if context and str(context).strip() else "None"
        
        # Special dataset handling
        if base_name == "SuperGLUE-RC":
            prompt = safe_format_prompt(prompt_template, Question=question, Answer="")
        elif base_name == "SuperGLUE-Wic":
            prompt = safe_format_prompt(prompt_template, Question=question, Context=context_str)
        elif base_name == "LiveCodeBench":
            # LiveCodeBench needs special handling from lcd dataset
            prompt = safe_format_prompt(prompt_template, Question=question)
        elif not options_str:
            prompt = safe_format_prompt(prompt_template, Context=context_str, Question=question)
        else:
            prompt = safe_format_prompt(prompt_template, Context=context_str, Question=question, Options=options_str)
        
        if len(prompt) > 10000:
            prompt = f"{prompt[:5000]}...{prompt[-5000:]}"
        
        formatted.append({
            "prompt_formatted": prompt,
            "global index": global_index,
        })
    
    return formatted

# Build and save
out_dir = "./dataset"
os.makedirs(out_dir, exist_ok=True)

for split in ["sub_10", "full"]:
    print(f"Building {split}...")
    data = build_formatted_prompts(split)
    filename = "router_data_10.json" if split == "sub_10" else "router_data.json"
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Wrote {len(data)} items → {path}")
    print(f"  Sample: {data[0]['prompt_formatted'][:120]}...")
    print()
