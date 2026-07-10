.PHONY: bootstrap bootstrap-core format lint typecheck test qa e0 gate-a gate-a-smoke

# Full research environment (torch included); required for gate-a and mypy.
bootstrap:
	uv sync --group dev --extra research --no-editable --reinstall-package longfeedback

# Minimal core environment matching the torch-free CI job.
bootstrap-core:
	uv sync --group dev --no-editable --reinstall-package longfeedback

format:
	uv run --no-sync ruff format .

lint:
	uv run --no-sync ruff check .

typecheck:
	uv run --no-sync mypy

test:
	uv run --no-sync pytest

qa: lint typecheck test
	uv run --no-sync ruff format --check .

e0:
	uv run --no-sync longfeedback experiment run e0

gate-a:
	uv run --no-sync longfeedback experiment run gate_a --config configs/experiments/gate_a.yaml

gate-a-smoke:
	uv run --no-sync longfeedback experiment run gate_a --config configs/experiments/gate_a_smoke.yaml
