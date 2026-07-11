.PHONY: bootstrap bootstrap-core format lint typecheck test qa e0 e1 e5 e6 gate-a gate-a-smoke gate-b data-lmsys data-kuairand data-wildchat e1-wildchat

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

data-lmsys:
	uv run --no-sync longfeedback data prepare lmsys --config configs/data/lmsys.yaml

data-wildchat:
	uv run --no-sync longfeedback data prepare wildchat --config configs/data/wildchat.yaml

data-kuairand:
	uv run --no-sync longfeedback data prepare kuairand --config configs/data/kuairand.yaml

e6:
	uv run --no-sync longfeedback experiment run e6 --config configs/experiments/e6.yaml

e1:
	uv run --no-sync longfeedback experiment run e1 --config configs/experiments/e1.yaml

e1-wildchat:
	uv run --no-sync longfeedback experiment run e1 --config configs/experiments/e1_wildchat.yaml

gate-b:
	uv run --no-sync longfeedback experiment run gate_b --config configs/experiments/gate_b.yaml

e5:
	uv run --no-sync longfeedback experiment run e5 --config configs/experiments/e5.yaml
