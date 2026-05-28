#!/usr/bin/env python3
"""
A3M Router adapter for RouterArena evaluation.

Routes queries to optimal models using A3M's multi-signal logic:
- Domain classification (code, math, creative, professional)
- Task-specialized model selection
- Hash-based deterministic distribution for simple MCQs
"""

import hashlib
import json
import re

# ============================================================
# MODEL PROFILES
# ============================================================

MODEL_PROFILES = {
    "gpt-4o-mini": {
        "provider": "OpenAI",
        "cost_per_1k_input": 0.15, "cost_per_1k_output": 0.60,
        "quality_score": 0.85,
        "strengths": ["generalist", "reasoning", "analysis"],
    },
    "claude-3-haiku-20240307": {
        "provider": "Anthropic",
        "cost_per_1k_input": 0.25, "cost_per_1k_output": 1.25,
        "quality_score": 0.83,
        "strengths": ["coding", "fast", "analysis"],
    },
    "gemini-2.5-flash": {
        "provider": "Google",
        "cost_per_1k_input": 0.15, "cost_per_1k_output": 0.60,
        "quality_score": 0.87,
        "strengths": ["multilingual", "long-context", "fast"],
    },
    "qwen/qwen3-235b-a22b-2507": {
        "provider": "Alibaba",
        "cost_per_1k_input": 0.90, "cost_per_1k_output": 0.90,
        "quality_score": 0.92,
        "strengths": ["reasoning", "multilingual", "analysis", "premium"],
    },
    "deepseek/deepseek-v4-flash": {
        "provider": "DeepSeek",
        "cost_per_1k_input": 0.15, "cost_per_1k_output": 0.60,
        "quality_score": 0.88,
        "strengths": ["coding", "fast", "budget"],
    },
}

# ============================================================
# QUERY CLASSIFICATION (same logic as A3M advancedRouter.js)
# ============================================================

CODE_PATTERNS = [
    r'\b(function|def|class|import|const|let|var|async|await|export)\b',
    r'\b(api|endpoint|rest|graphql|middleware|route)\b',
    r'\b(sql|database|query|index|migration|schema)\b',
    r'\b(docker|kubernetes|ci/cd|deploy|pipeline)\b',
    r'\b(react|vue|angular|next\.js|node\.js|python|django)\b',
    r'\b(bug|debug|error|exception|crash|traceback)\b',
    r'\b(algorithm|complexity|optimize|refactor)\b',
    r'```|def\s+\w+\s*\(|class\s+\w+',
]

MATH_PATTERNS = [
    r'\b(calculate|compute|equation|formula|solve|proof)\b',
    r'\b(integral|derivative|matrix|vector|eigen)\b',
    r'\b(probability|statistic|regression|correlation)\b',
    r'\d+\s*[\+\-\*\/]\s*\d+',
    r'\b(theorem|lemma|axiom|conjecture)\b',
]

CREATIVE_PATTERNS = [
    r'\b(write|compose|create|design|generate).*(story|poem|song|article|blog)\b',
    r'\b(brainstorm|idea|creative|innovative)\b',
    r'\b(narrative|plot|character|dialogue|scene)\b',
]

PROFESSIONAL_DOMAINS = {
    "medical": ["clinical", "medical", "pharmaceutical", "diagnosis", "treatment",
                 "patient", "surgical", "disease", "drug", "therapy"],
    "legal": ["legal", "law", "contract", "regulation", "compliance", "court",
              "attorney", "patent", "copyright", "statute"],
    "finance": ["financial", "valuation", "revenue", "portfolio", "investment",
                "tax", "accounting", "fiscal", "audit", "monetary"],
}


def classify_query(prompt: str) -> dict:
    """Extract query features."""
    lower = prompt.lower()
    words = prompt.split()

    features = {
        "word_count": len(words),
        "code_score": min(sum(bool(re.search(p, prompt, re.I)) for p in CODE_PATTERNS) * 0.15, 1.0),
        "math_score": min(sum(bool(re.search(p, prompt, re.I)) for p in MATH_PATTERNS) * 0.2, 1.0),
        "creative_score": min(sum(bool(re.search(p, prompt, re.I)) for p in CREATIVE_PATTERNS) * 0.15, 1.0),
        "domain": "general",
        "is_mcq": "Options:" in prompt and len(words) < 80,
    }

    # Domain detection
    for domain, keywords in PROFESSIONAL_DOMAINS.items():
        if sum(kw in lower for kw in keywords) >= 2:
            features["domain"] = domain
            break

    # Complexity
    wc = features["word_count"]
    if wc < 30:
        features["complexity"] = 0.3
    elif wc > 200:
        features["complexity"] = 0.9
    elif wc > 100:
        features["complexity"] = 0.7
    elif wc > 50:
        features["complexity"] = 0.5
    else:
        features["complexity"] = 0.4

    return features


# ============================================================
# ROUTING ENGINE
# ============================================================

def select_model(query: str, available_models: list) -> str:
    """Select the best model using A3M-style routing."""
    features = classify_query(query)
    query_hash = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)

    model_scores = {}
    for model_name in available_models:
        profile = MODEL_PROFILES.get(model_name, {})
        if not profile:
            continue
        provider = profile.get("provider")
        quality = profile.get("quality_score", 0.80)
        cost = profile.get("cost_per_1k_input", 0.0)
        score = quality * 1.0

        # Task specialization
        if features["code_score"] > 0.1:
            if provider == "DeepSeek":
                score += 0.6
            elif provider == "Anthropic":
                score += 0.4
        if features["math_score"] > 0.1:
            if provider == "Alibaba":
                score += 0.5
            elif provider == "Anthropic":
                score += 0.3
        if features["creative_score"] > 0.1:
            if provider == "Anthropic":
                score += 0.5
        if features["domain"] != "general":
            if provider == "Alibaba":
                score += 0.5
            elif provider == "Anthropic":
                score += 0.3

        # Short MCQ distribution: hash-based even spread across models
        if features["is_mcq"]:
            bucket = query_hash % len(available_models)
            if available_models[bucket] == model_name:
                score += 1.5  # moderate boost for assigned slot
            else:
                score -= 0.5  # slight penalty for non-assigned

        # Complexity
        if features["complexity"] > 0.5:
            if provider == "Alibaba":
                score += 0.6
            elif provider == "OpenAI":
                score += 0.4
        if features["complexity"] < 0.3:
            if provider == "DeepSeek":
                score += 0.3

        # Cost efficiency
        if cost > 0:
            score += quality / (cost + 0.01) * 0.03

        model_scores[model_name] = score + (query_hash % 100) * 0.001

    return max(model_scores, key=model_scores.get) if model_scores else available_models[0]


# ============================================================
# RouterArena BaseRouter interface
# ============================================================

from router_inference.router.base_router import BaseRouter


class A3MRouter(BaseRouter):
    """A3M Router adapter for RouterArena evaluation."""

    def __init__(self, router_name: str):
        super().__init__(router_name)

    def _get_prediction(self, query: str) -> str:
        return select_model(query, self.models)
