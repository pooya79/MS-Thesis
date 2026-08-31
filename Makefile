.PHONY: help run test

help:
	@echo "Available targets:"
	@echo "  make run   - Start FastAPI development server"
	@echo "  make test  - Run test suite"

run:
	uv run uvicorn server.app.main:app --reload --host 0.0.0.0 --port 8001

test:
	uv run pytest -q
