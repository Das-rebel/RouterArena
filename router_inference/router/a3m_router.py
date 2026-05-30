#!/usr/bin/env python3
"""A3M Router - distributes across 4 free providers."""
from router_inference.router.base_router import BaseRouter
import hashlib

class A3MRouter(BaseRouter):
    MODELS = [
        "deepseek-chat",
        "meta-llama_llama-3.3-70b-instruct",
        "meta/llama-3.3-70b-instruct",
        "mistralai/ministral-3-14b-2512",
    ]
    def __init__(self, router_name: str) -> None:
        super().__init__(router_name)
    def _get_prediction(self, query: str) -> str:
        h = int(hashlib.md5(query.encode()).hexdigest(), 16)
        return self.MODELS[h % 4]
