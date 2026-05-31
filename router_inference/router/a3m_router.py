#!/usr/bin/env python3
from router_inference.router.base_router import BaseRouter
import hashlib

class A3MRouter(BaseRouter):
    MODELS = [
        "deepseek-chat","deepseek-chat","deepseek-chat",                           # 30%
        "meta/llama-3.3-70b-instruct","meta/llama-3.3-70b-instruct","meta/llama-3.3-70b-instruct","meta/llama-3.3-70b-instruct",  # 40%
        "deepseek-chat","deepseek-chat","deepseek-chat",                           # 30%
    ]
    def __init__(self, rn): super().__init__(rn)
    def _get_prediction(self, q):
        return self.MODELS[int(hashlib.md5(q.encode()).hexdigest(),16)%len(self.MODELS)]
