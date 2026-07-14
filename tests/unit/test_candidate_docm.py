"""Candidate DOCM: policy-centered dueling Q contract and head behavior."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from longfeedback.models.candidate_data import CandidateSequenceDataset
from longfeedback.models.candidate_docm import (
    CANDIDATE_VARIANTS,
    CandidateDelayedOutcomeCreditModel,
    CandidateLossWeights,
    CandidateTrainingSettings,
    candidate_variant_spec,
    policy_centered_q,
)
from longfeedback.models.encoders import CausalTransformerEncoder, EncoderArchitecture

_ARCHITECTURE = EncoderArchitecture(d_model=16, n_layers=1, n_heads=2)
_EPS = 1.0e-4


def _random_inputs(
    seed: int, batch: int = 5, candidates: int = 6, *, extreme: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    scale = 50.0 if extreme else 2.0
    u = (torch.rand(batch, candidates, generator=generator) - 0.5) * scale
    v_logit = (torch.rand(batch, generator=generator) - 0.5) * scale
    scale_logit = (torch.rand(batch, generator=generator) - 0.5) * scale
    mask = torch.rand(batch, candidates, generator=generator) > 0.25
    mask[:, 0] = True
    raw = torch.rand(batch, candidates, generator=generator) * mask
    probabilities = raw / raw.sum(dim=-1, keepdim=True)
    return u, v_logit, scale_logit, probabilities, mask


@pytest.mark.parametrize("extreme", [False, True])
def test_policy_centering_identity_holds(extreme: bool) -> None:
    u, v_logit, scale_logit, probabilities, mask = _random_inputs(0, extreme=extreme)
    q, v, _ = policy_centered_q(
        u.double(),
        v_logit.double(),
        scale_logit.double(),
        probabilities.double(),
        mask,
        epsilon=_EPS,
    )
    center = (torch.where(mask, probabilities.double(), torch.zeros_like(q)) * q).sum(-1)
    assert torch.max(torch.abs(center - v)) < 1.0e-6
    assert torch.all(q[mask] >= 0.0)
    assert torch.all(q[mask] <= 1.0)
    # Wherever V itself sits inside the epsilon corridor, every Q must too;
    # a saturated V collapses the scale to zero and Q equals V exactly.
    for row in range(q.shape[0]):
        row_q = q[row][mask[row]]
        if _EPS <= float(v[row]) <= 1.0 - _EPS:
            assert torch.all(row_q >= _EPS - 1.0e-9)
            assert torch.all(row_q <= 1.0 - _EPS + 1.0e-9)
        else:
            assert torch.allclose(row_q, v[row].expand_as(row_q))


def test_constant_shift_in_raw_advantages_does_not_change_q() -> None:
    u, v_logit, scale_logit, probabilities, mask = _random_inputs(1)
    q, _, _ = policy_centered_q(u, v_logit, scale_logit, probabilities, mask, epsilon=_EPS)
    shifted, _, _ = policy_centered_q(
        u + 3.7, v_logit, scale_logit, probabilities, mask, epsilon=_EPS
    )
    assert torch.allclose(q[mask], shifted[mask], atol=1.0e-5)


def test_candidate_reordering_leaves_q_unchanged() -> None:
    u, v_logit, scale_logit, probabilities, mask = _random_inputs(2)
    permutation = torch.randperm(u.shape[1], generator=torch.Generator().manual_seed(3))
    q, _, _ = policy_centered_q(u, v_logit, scale_logit, probabilities, mask, epsilon=_EPS)
    q_permuted, _, _ = policy_centered_q(
        u[:, permutation],
        v_logit,
        scale_logit,
        probabilities[:, permutation],
        mask[:, permutation],
        epsilon=_EPS,
    )
    assert torch.allclose(q[:, permutation], q_permuted, atol=1.0e-6)


def test_all_zero_directions_collapse_to_v() -> None:
    batch, candidates = 3, 4
    u = torch.zeros(batch, candidates)
    v_logit = torch.tensor([0.0, 2.0, -2.0])
    scale_logit = torch.zeros(batch)
    probabilities = torch.full((batch, candidates), 0.25)
    mask = torch.ones(batch, candidates, dtype=torch.bool)
    q, v, scale = policy_centered_q(u, v_logit, scale_logit, probabilities, mask, epsilon=_EPS)
    assert torch.allclose(q, v.unsqueeze(-1).expand_as(q))
    assert torch.all(scale == 0.0)


def _tiny_dataset(
    *, seed: int = 0, candidates: int = 3, pad_extra: int = 0
) -> CandidateSequenceDataset:
    rng = np.random.default_rng(seed)
    episodes, horizon, dim = 4, 3, 8
    total = candidates + pad_extra
    step_mask = np.ones((episodes, horizon), dtype=np.bool_)
    step_mask[0, 2] = False
    candidate_mask = np.zeros((episodes, total), dtype=np.bool_)
    candidate_mask[:, :candidates] = True
    probabilities = np.zeros((episodes, total), dtype=np.float64)
    probabilities[:, :candidates] = 1.0 / candidates
    q_count = np.zeros((episodes, total), dtype=np.int64)
    q_count[:, : candidates - 1] = 4
    q_success = np.minimum(rng.integers(0, 5, size=(episodes, total)), q_count).astype(np.int64)
    candidate_embeddings = np.zeros((episodes, total, dim), dtype=np.float64)
    candidate_embeddings[:, :candidates] = rng.normal(size=(episodes, candidates, dim))
    return CandidateSequenceDataset(
        state_embeddings=rng.normal(size=(episodes, horizon, dim)),
        action_embeddings=rng.normal(size=(episodes, horizon, dim)),
        step_mask=step_mask,
        outcomes=rng.integers(0, 2, size=episodes).astype(np.float64),
        branch_episode=np.arange(episodes, dtype=np.int64),
        branch_step=np.zeros(episodes, dtype=np.int64) + 1,
        candidate_embeddings=candidate_embeddings,
        candidate_mask=candidate_mask,
        candidate_probabilities=probabilities,
        q_success=q_success,
        q_count=q_count,
        v_success=rng.integers(0, 5, size=episodes).astype(np.int64),
        v_count=np.full(episodes, 4, dtype=np.int64),
        forced_done=np.zeros((episodes, total), dtype=np.bool_),
        forced_success=np.zeros((episodes, total), dtype=np.float64),
        child_v_success=np.zeros((episodes, total), dtype=np.int64),
        child_v_count=np.zeros((episodes, total), dtype=np.int64),
        child_state_embeddings=rng.normal(size=(episodes, total, dim)),
    )


def _model(seed: int = 0, **kwargs: object) -> CandidateDelayedOutcomeCreditModel:
    weights, parameterization = candidate_variant_spec("docm_dueling_credit")
    return CandidateDelayedOutcomeCreditModel(
        state_dim=8,
        action_dim=8,
        max_horizon=3,
        architecture=_ARCHITECTURE,
        loss_weights=weights,
        action_value_parameterization=parameterization,
        action_mlp_hidden=8,
        target_network_ema=0.5,
        seed=seed,
        **kwargs,  # type: ignore[arg-type]
    )


def _pad_candidates(dataset: CandidateSequenceDataset, extra: int) -> CandidateSequenceDataset:
    def widen(array: np.ndarray, fill: float = 0.0) -> np.ndarray:
        pad_shape = (array.shape[0], extra, *array.shape[2:])
        return np.concatenate([array, np.full(pad_shape, fill, dtype=array.dtype)], axis=1)

    fields = dict(dataset.__dict__)
    for name in (
        "candidate_embeddings",
        "candidate_mask",
        "candidate_probabilities",
        "q_success",
        "q_count",
        "forced_done",
        "forced_success",
        "child_v_success",
        "child_v_count",
        "child_state_embeddings",
    ):
        fields[name] = widen(fields[name])
    return CandidateSequenceDataset(**fields)


def test_padded_candidates_do_not_change_real_predictions() -> None:
    base = _tiny_dataset()
    padded = _pad_candidates(base, 2)
    model = _model()
    q_base = model.predict_branch_q(base)
    q_padded = model.predict_branch_q(padded)
    assert np.allclose(q_base, q_padded[:, :3], atol=1.0e-6)
    assert np.all(q_padded[:, 3:] == 0.0)


def test_missing_full_distribution_is_a_hard_error() -> None:
    dataset = _tiny_dataset()
    broken = dict(dataset.__dict__)
    probabilities = dataset.candidate_probabilities.copy()
    probabilities[0, 0] = 0.0
    broken["candidate_probabilities"] = probabilities
    with pytest.raises(ValueError, match="full admissible set"):
        CandidateSequenceDataset(**broken)


def test_parameter_counts_identical_across_variants() -> None:
    counts = set()
    for variant in CANDIDATE_VARIANTS:
        weights, parameterization = candidate_variant_spec(variant)
        model = CandidateDelayedOutcomeCreditModel(
            state_dim=8,
            action_dim=8,
            max_horizon=3,
            architecture=_ARCHITECTURE,
            loss_weights=weights,
            action_value_parameterization=parameterization,
            action_mlp_hidden=8,
            seed=0,
        )
        counts.add(model.parameter_count())
    assert len(counts) == 1


def test_binomial_nll_matches_hand_computation() -> None:
    model = _model()
    probabilities = torch.tensor([[0.5, 0.25]])
    successes = torch.tensor([[2.0, 1.0]])
    counts = torch.tensor([[4.0, 0.0]])
    loss = model._binomial_nll(probabilities, successes, counts)
    expected = -(2.0 * np.log(0.5) + 2.0 * np.log(0.5)) / 4.0
    assert float(loss) == pytest.approx(expected, rel=1.0e-6)


def test_fit_keeps_policy_centering_identity() -> None:
    dataset = _tiny_dataset()
    model = _model()
    model.fit(dataset, training=CandidateTrainingSettings(epochs=2, batch_size=8))
    assert model.policy_centering_residual(dataset) < 1.0e-6


def test_appended_scoring_ignores_post_action_information() -> None:
    dataset = _tiny_dataset()
    modified = dict(dataset.__dict__)
    modified["child_state_embeddings"] = dataset.child_state_embeddings + 5.0
    changed = CandidateSequenceDataset(**modified)
    model = _model()
    for head in ("outcome", "value"):
        base_scores = model.predict_appended_candidate_scores(dataset, head=head)
        changed_scores = model.predict_appended_candidate_scores(changed, head=head)
        assert np.allclose(base_scores, changed_scores)


def test_hindsight_weight_rejected_in_cpu_milestone() -> None:
    with pytest.raises(NotImplementedError):
        CandidateLossWeights(outcome=1.0, hindsight_aux=0.5)


def test_encoder_padding_mask_preserves_no_mask_behavior() -> None:
    encoder = CausalTransformerEncoder(input_dim=4, max_length=6, architecture=_ARCHITECTURE)
    tokens = torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(0))
    encoder.eval()
    with torch.no_grad():
        unmasked = encoder(tokens)
        with_empty_mask = encoder(tokens, padding_mask=torch.zeros(2, 5, dtype=torch.bool))
    assert torch.allclose(unmasked, with_empty_mask, atol=1.0e-5)


def test_right_padding_cannot_change_earlier_positions() -> None:
    encoder = CausalTransformerEncoder(input_dim=4, max_length=8, architecture=_ARCHITECTURE)
    generator = torch.Generator().manual_seed(1)
    tokens = torch.randn(1, 4, 4, generator=generator)
    padded = torch.cat([tokens, torch.randn(1, 2, 4, generator=generator)], dim=1)
    encoder.eval()
    with torch.no_grad():
        short = encoder(tokens)
        long = encoder(padded)
    assert torch.allclose(short, long[:, :4, :], atol=1.0e-5)
