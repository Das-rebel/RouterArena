# Copyright 2024-present RouteWorks, Inc.
# Licensed under the Apache License, Version 2.0
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
from router_inference.router.base_router import BaseRouter
import hashlib

class A3MRouter(BaseRouter):
    MODELS = ["deepseek-chat"] * 10  # 100%
    def __init__(self, rn): super().__init__(rn)
    def _get_prediction(self, q):
        return self.MODELS[int(hashlib.md5(q.encode()).hexdigest(),16)%len(self.MODELS)]
