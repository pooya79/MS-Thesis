.PHONY: help run test

help:
	@echo "Available targets:"
	@echo "  make run   - Start FastAPI development server using .env and installed packages (no dependency sync)"
	@echo "  make test  - Run test suite"

run:
	uv run --no-sync --env-file .env uvicorn server.app.main:app --reload --host 0.0.0.0 --port 8001

test:
	uv run pytest -q
