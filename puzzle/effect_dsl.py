from typing import List, Dict, Any, Optional, TypedDict

class TargetSpec(TypedDict, total=False):
    owner: str
    zone: str
    count: int

class EffectStep(TypedDict, total=False):
    op: str
    target: Optional[TargetSpec]
    amount: Optional[int]
    until: Optional[str]

class EffectProgram(TypedDict, total=False):
    effect: List[EffectStep]

