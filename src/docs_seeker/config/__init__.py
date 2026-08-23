"""docs-seeker - 配置模块"""
from pathlib import Path

import yaml

from docs_seeker.config.settings import settings

_CONFIG_DIR = Path(__file__).resolve().parent


def _load_yaml(name: str) -> dict:
    """读取 config 目录下的 yaml 配置，缺失/解析失败时返回空 dict"""
    path = _CONFIG_DIR / name
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# Prompt 模板（generator/query_decomposer 等使用）
prompts = _load_yaml("prompts.yaml")

# 检索策略配置（RRF 权重/k、单路召回参数等）
retrieval_config = _load_yaml("retrieval.yaml")

__all__ = ["settings", "prompts", "retrieval_config"]
