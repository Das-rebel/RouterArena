#!/usr/bin/env python3
"""Build formatted prompts using the EXACT same logic as prep_datasets.py."""
import sys, os, json

# Setup paths like prep_datasets.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = SCRIPT_DIR  # script is in repo root
sys.path.insert(0, REPO_DIR)

from datasets import load_dataset, load_from_disk
from typing import Dict, Any, List

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

def build_formatted_prompts_from_router_eval_benchmark(split_name: str) -> List[Dict[str, Any]]:
    ds = load_dataset("RouteWorks/RouterArena")[split_name]

    # Load LiveCodeBench from disk (try, may not exist)
    lcd_global_idx_map: Dict[str, Dict[str, Any]] = {}
    lcd_index_map: Dict[str, Dict[str, Any]] = {}
    lcd_prompt_prefix_map: Dict[str, Dict[str, Any]] = {}
    try:
        lcd = load_from_disk(os.path.join(SCRIPT_DIR, "dataset", "livecodebench"))
        lcd_dataset_list = lcd.to_list()
        for item in lcd_dataset_list:
            gidx_key = str(item.get("global_idx")) if item.get("global_idx") is not None else None
            if gidx_key is not None:
                lcd_global_idx_map[gidx_key] = item
            idx_key = str(item.get("_index")) if item.get("_index") is not None else None
            if idx_key is not None:
                lcd_index_map[idx_key] = item
            prompt_text = item.get("prompt") or ""
            if isinstance(prompt_text, str) and prompt_text:
                lcd_prompt_prefix_map[prompt_text[:120]] = item
    except Exception as e:
        print(f"[prep] Warning: could not load livecodebench: {e}")

    # Load configs
    config_dir = os.path.join(REPO_DIR, "config", "eval_config", "zero-shot")
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
    dataset_configs: Dict[str, Dict[str, Any]] = {}
    for dataset_name in dataset_names:
        cfg_path = os.path.join(config_dir, f"{dataset_name}.json")
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                dataset_configs[dataset_name] = cfg.get("eval_params", {})
        except FileNotFoundError:
            continue

    formatted: List[Dict[str, Any]] = []
    for row in ds:
        global_index = (row.get("Global Index") or row.get("global_index") or row.get("global index"))
        
        if not row.get("Dataset name"):
            global_index_parts = row["Global Index"].split("_")
            if global_index_parts[0] == "Ethics" and len(global_index_parts) >= 2:
                dataset_name_full = f"{global_index_parts[0]}_{global_index_parts[1]}"
            elif global_index_parts[0] == "ChessInstruct" and len(global_index_parts) >= 2:
                dataset_name_full = f"{global_index_parts[0]}_{global_index_parts[1]}"
            else:
                dataset_name_full = global_index_parts[0]
        else:
            dataset_name_full = row.get("Dataset name")

        options_list = row.get("Options")
        has_options = options_list is not None and len(options_list) > 0

        if "Ethics" in dataset_name_full:
            base_dataset_name = dataset_name_full
        elif "ChessInstruct" in dataset_name_full:
            if has_options:
                base_dataset_name = "ChessInstruct_mcq"
            else:
                base_dataset_name = "ChessInstruct"
        else:
            base_dataset_name = str(dataset_name_full).split("_", 1)[0]

        assert base_dataset_name in dataset_configs, f"No config for {base_dataset_name}"
        eval_params = dataset_configs[base_dataset_name]

        # Build options string
        if has_options:
            labels = [chr(65 + i) for i in range(len(options_list))]
            options_str = "\n".join(f"{label}. {opt}" for label, opt in zip(labels, options_list))
        else:
            options_str = ""
            options_list = []

        question = row.get("Question") or ""
        context_val = row.get("Context") or ""
        context_for_prompt = context_val if context_val != "" else "None"

        if base_dataset_name == "LiveCodeBench":
            # Use default template if LiveCodeBench dataset not available
            lcd_row = lcd_global_idx_map.get(str(global_index))
            if lcd_row is None:
                idx = str(global_index).split("_")[-1]
                lcd_row = lcd_index_map.get(idx)
            if lcd_row is None:
                q_prefix = (question or "")[:120]
                lcd_row = lcd_prompt_prefix_map.get(q_prefix)
            if lcd_row is None:
                # Fallback: use generic LCB prompt
                prompt_formatted = safe_format_prompt(
                    eval_params.get("prompt", "{Question}"),
                    Question=question
                )
            else:
                prompt = (eval_params.get("is_stdin_prompt") if lcd_row.get("is_stdin") else eval_params.get("not_is_stdin_prompt"))
                prompt_formatted = safe_format_prompt(prompt or "{Question}", Question=question)
        elif base_dataset_name == "SuperGLUE-RC":
            prompt_formatted = safe_format_prompt(eval_params.get("prompt", "{Question}"), Question=question, Answer="")
        elif base_dataset_name == "SuperGLUE-Wic":
            prompt_formatted = safe_format_prompt(eval_params.get("prompt", "{Question}"), Question=question, Context=context_val)
        elif not options_list:
            prompt_formatted = safe_format_prompt(eval_params.get("prompt", "{Question}"), Context=context_for_prompt, Question=question)
        else:
            prompt_formatted = safe_format_prompt(eval_params.get("prompt", "{Question}"), Context=context_for_prompt, Question=question, Options=options_str)

        if len(prompt_formatted) > 10000:
            prompt_formatted = f"{prompt_formatted[:5000]}...{prompt_formatted[-5000:]}"

        assert len(prompt_formatted) > 0, f"Prompt empty for {dataset_name_full}"

        formatted.append({
            "prompt_formatted": prompt_formatted,
            "global index": global_index,
        })

    return formatted

if __name__ == "__main__":
    out_dir = os.path.join(SCRIPT_DIR, "dataset")
    os.makedirs(out_dir, exist_ok=True)

    for split in ["sub_10", "full"]:
        print(f"Building {split}...")
        data = build_formatted_prompts_from_router_eval_benchmark(split)
        filename = "router_data_10.json" if split == "sub_10" else "router_data.json"
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Wrote {len(data)} items → {path}")
        print(f"  Sample: {data[0]['prompt_formatted'][:150]}...")
        print()
