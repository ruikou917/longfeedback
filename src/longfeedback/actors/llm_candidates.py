"""Hugging Face causal-LM candidate scorer (frozen or LoRA-trainable).

The score of a candidate command is its mean token log-probability given the
prompt; length normalization is part of the policy definition. This module is
imported lazily by the runners and requires the (GPU-oriented) ``transformers``
and optional ``peft`` packages; it is not exercised by the CPU test suite and
must be validated in the cloud-GPU step before any real E11/E12 run.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from longfeedback.actors.base import (
    PolicyDecision,
    PolicyScores,
    canonical_candidates,
    sample_from_scores,
    softmax_scores,
)


@dataclass(frozen=True, slots=True)
class LlmActorSettings:
    model_id: str
    model_revision: str
    tokenizer_revision: str
    temperature: float = 1.0
    max_prompt_tokens: int = 2048
    device: str = "cpu"
    torch_dtype: str = "float32"
    lora_rank: int | None = None
    lora_alpha: float | None = None
    lora_target_modules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_id or not self.model_revision or not self.tokenizer_revision:
            raise ValueError("model_id, model_revision, and tokenizer_revision are required")
        if self.model_revision in ("main", "master") or self.tokenizer_revision in (
            "main",
            "master",
        ):
            raise ValueError("floating revisions are not reproducible; pin a commit hash")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")


@dataclass
class LlmCandidatePolicy:
    """Scores every admissible command with one forward pass per candidate."""

    settings: LlmActorSettings
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model_id, revision=self.settings.tokenizer_revision
        )
        dtype = getattr(torch, self.settings.torch_dtype)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.settings.model_id,
            revision=self.settings.model_revision,
            torch_dtype=dtype,
        ).to(self.settings.device)
        if self.settings.lora_rank is not None:
            from peft import LoraConfig, get_peft_model

            lora = LoraConfig(
                r=self.settings.lora_rank,
                lora_alpha=self.settings.lora_alpha or 2 * self.settings.lora_rank,
                target_modules=list(self.settings.lora_target_modules) or None,
            )
            self._model = get_peft_model(self._model, lora)
        self._model.eval()

    @property
    def policy_id(self) -> str:
        digest = hashlib.sha256(f"{self.settings.model_id}@{self.settings.model_revision}".encode())
        if self._model is not None and self.settings.lora_rank is not None:
            import numpy as np

            for name, parameter in sorted(self._model.named_parameters()):
                if "lora" in name:
                    digest.update(name.encode("utf-8"))
                    digest.update(
                        np.ascontiguousarray(
                            parameter.detach().float().cpu().numpy(), dtype=np.float32
                        ).tobytes()
                    )
        return f"llm:{digest.hexdigest()[:16]}"

    def _mean_token_logprob(self, prompt_ids: Any, candidate_ids: Any) -> float:
        import torch

        input_ids = torch.cat([prompt_ids, candidate_ids], dim=-1).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(input_ids.to(self.settings.device)).logits[0]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        start = prompt_ids.shape[-1]
        total = 0.0
        for position in range(candidate_ids.shape[-1]):
            token = int(candidate_ids[position])
            total += float(log_probs[start + position - 1, token])
        return total / max(1, int(candidate_ids.shape[-1]))

    def score(self, prompt: str, candidates: Sequence[str]) -> PolicyScores:
        self._ensure_loaded()
        import torch

        canonical = canonical_candidates(candidates)
        prompt_ids = torch.tensor(
            self._tokenizer(prompt, truncation=True, max_length=self.settings.max_prompt_tokens)[
                "input_ids"
            ],
            dtype=torch.long,
        )
        raw_scores: list[float] = []
        token_counts: list[int] = []
        forward_tokens = 0
        for candidate in canonical:
            candidate_ids = torch.tensor(
                self._tokenizer(" " + candidate, add_special_tokens=False)["input_ids"],
                dtype=torch.long,
            )
            raw_scores.append(self._mean_token_logprob(prompt_ids, candidate_ids))
            token_counts.append(int(candidate_ids.shape[-1]))
            forward_tokens += int(prompt_ids.shape[-1]) + int(candidate_ids.shape[-1])
        return softmax_scores(
            canonical,
            raw_scores,
            token_counts,
            forward_tokens=forward_tokens,
            temperature=self.settings.temperature,
        )

    def sample(self, scores: PolicyScores, *, random_value: float) -> PolicyDecision:
        return sample_from_scores(scores, random_value=random_value)
