#!/usr/bin/env python3
from router_inference.router.base_router import BaseRouter

class A3MRouter(BaseRouter):
    def __init__(self, router_name: str) -> None:
        super().__init__(router_name)
    def _get_prediction(self, query: str) -> str:
        return "mistralai/ministral-3-14b-2512"
