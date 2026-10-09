"""Runtime settings, read from environment variables (prefix ``PLAGAI_``) or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PLAGAI_", env_file=".env", extra="ignore")

    # --- storage -----------------------------------------------------------
    database_url: str = "sqlite:///./plagai.db"
    corpus_dir: str = "./artifacts/corpus"

    # --- AI detector -------------------------------------------------------
    # Directory written by `plagai train-ai` (contains the HF model + ensemble.json).
    ai_model_dir: str = "./artifacts/ai-detector"
    # Binoculars observer/performer pair. They must share a tokenizer.
    # The paper used falcon-7b / falcon-7b-instruct; the Qwen pair fits on CPU or a small GPU.
    binoculars_observer: str = "Qwen/Qwen2.5-1.5B"
    binoculars_performer: str = "Qwen/Qwen2.5-1.5B-Instruct"
    enable_binoculars: bool = False
    ai_max_length: int = 512
    device: str = "auto"  # auto | cpu | cuda | mps

    # --- plagiarism detector ----------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    plag_paraphrase_threshold: float = 0.80
    plag_exact_overlap: float = 0.80

    # --- API / billing -----------------------------------------------------
    admin_token: str = "change-me"
    words_per_credit: int = 100
    min_credits_per_call: int = 1
    max_words_per_request: int = 25_000
    signup_free_credits: int = 0
    rate_limit_per_minute: int = 60
    # Lets the API boot without trained weights (for development and tests):
    # the AI endpoint then uses stylometry only.
    allow_untrained: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
