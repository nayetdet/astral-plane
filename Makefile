.PHONY: run lint format typecheck

run:
	PYTHONUNBUFFERED=1 uv run python -m astral_plane

lint:
	ruff check .

format:
	ruff format .

typecheck:
	uv run mypy src
