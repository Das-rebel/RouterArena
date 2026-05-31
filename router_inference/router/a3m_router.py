#!/usr/bin/env python3
from router_inference.router.base_router import BaseRouter
from typing import Optional

class A3MRouter(BaseRouter):
    ROUTING = {
        'MMLUPro_computer science': 'deepseek-chat',
        'MMLUPro_history': 'deepseek-chat',
        'MMLUPro_engineering': 'deepseek-chat',
        'MMLUPro_health': 'deepseek-chat',
        'MMLUPro_math': 'deepseek-chat',
        'MMLUPro_business': 'deepseek-chat',
        'MMLUPro_chemistry': 'deepseek-chat',
        'MMLUPro_biology': 'deepseek-chat',
        'MMLUPro_physics': 'deepseek-chat',
        'ArcMMLU': 'deepseek-chat',
        'LiveCodeBench': 'deepseek-chat',
        'QANTA': 'deepseek-chat',
        'OpenTDB': 'deepseek-chat',
        'MathQA': 'deepseek-chat',
        'MusicTheoryBench': 'deepseek-chat',
        'MMLUPro_psychology': 'deepseek-chat',
        'MMLUPro_economics': 'deepseek-chat',
        'MMLUPro_philosophy': 'deepseek-chat',
        'NarrativeQA': 'deepseek-chat',
        'Ethics': 'deepseek-chat',
        'PubMedQA': 'mistralai/ministral-3-14b-2512',
        'GeoBench': 'mistralai/ministral-3-14b-2512',
        'MedMCQA': 'mistralai/ministral-3-14b-2512',
        'MMLUPro_law': 'mistralai/ministral-3-14b-2512',
    }
    MODELS = list(set(ROUTING.values()))
    
    def _get_prediction(self, query: str, global_index: Optional[str] = None) -> str:
        if global_index:
            for pattern, model in sorted(self.ROUTING.items(), key=lambda x: -len(x[0])):
                if global_index.startswith(pattern):
                    return model
        return 'deepseek-chat'
