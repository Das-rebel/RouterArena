#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
A3M Router for RouterArena evaluation.
Routes to cheapest reliable model via OpenRouter.
"""

from typing import Any, Dict, List
from router_inference.router.base_router import BaseRouter

# Priority: cheapest non-reasoning model first
RELIABLE_MODELS = {
    "openai/gpt-4o-mini": {
        "cost_per_1k": 0.15,
        "strength": "fast-cheap-generalist"
    },
    "deepseek/deepseek-v4-flash": {
        "cost_per_1k": 0.15,
        "strength": "coding-reasoning"
    },
    "qwen/qwen3-235b-a22b-2507": {
        "cost_per_1k": 0.90,
        "strength": "reasoning-premium"
    },
}

PRIMARY = "openai/gpt-4o-mini"
FALLBACKS = ["deepseek/deepseek-v4-flash", "qwen/qwen3-235b-a22b-2507"]

class A3MRouter(BaseRouter):
    """Routes to cheapest working model, falls back to alternatives."""
    
    def __init__(self, router_name: str) -> None:
        super().__init__(router_name)
    
    def _get_prediction(self, query: str) -> str:
        """Select the optimal model for this query."""
        available = [m for m in self.models if m in RELIABLE_MODELS]
        
        if not available:
            return PRIMARY
        
        if PRIMARY in available:
            return PRIMARY
        
        for fb in FALLBACKS:
            if fb in available:
                return fb
        
        return available[0]
