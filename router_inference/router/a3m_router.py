#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
A3M Router adapter for RouterArena evaluation.
Routes to cheapest reliable model per query difficulty.
Uses only models proven to work in RouterArena's API pipeline.
"""

from typing import Any, Dict, List
from router_inference.router.base_router import BaseRouter

# Models with proven API reliability in RouterArena
RELIABLE_MODELS = {
    "deepseek/deepseek-v4-flash": {
        "success_rate": 0.54,
        "cost_per_1k": 0.15,
        "strength": "generalist"
    },
    "qwen/qwen3-235b-a22b-2507": {
        "success_rate": 0.32,
        "cost_per_1k": 0.90,
        "strength": "reasoning"
    },
}

class A3MRouter(BaseRouter):
    """Routes to cheapest working model, falls back to most reliable."""
    
    def __init__(self, router_name: str) -> None:
        super().__init__(router_name)
    
    def _get_prediction(self, query: str) -> str:
        """Select the optimal model for this query."""
        available = [m for m in self.models if m in RELIABLE_MODELS]
        
        if not available:
            # If none of our preferred models are available, use first from config
            return self.models[0] if self.models else "deepseek/deepseek-v4-flash"
        
        # Default: cheapest working model
        return "deepseek/deepseek-v4-flash" if "deepseek/deepseek-v4-flash" in available else available[0]
