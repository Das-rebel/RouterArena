#!/usr/bin/env python3
"""A3M Router - distributes across working free models."""
from router_inference.router.base_router import BaseRouter
import hashlib

class A3MRouter(BaseRouter):
    def __init__(self, router_name: str) -> None:
        super().__init__(router_name)
    
    def _get_prediction(self, query: str) -> str:
        # Hash-based distribution: even → mistral, odd → zhipu
        h = int(hashlib.md5(query.encode()).hexdigest(), 16)
        return "mistralai/ministral-3-14b-2512" if h % 2 == 0 else "glm-4.6"
