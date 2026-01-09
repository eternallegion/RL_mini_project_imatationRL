# utils.py
from __future__ import annotations
import os
import logging

def setup_root_logger():
    os.makedirs("logs", exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # 중복 방지
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        sh.setFormatter(fmt)
        root.addHandler(sh)
