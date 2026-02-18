.PHONY: help run test migrate-up migrate-down

help:
	@echo "Available targets:"
	@echo "  make run   - Start FastAPI development server"
	@echo "  make test  - Run test suite"
	@echo "  make migrate-up   - Apply all pending Alembic migrations"
	@echo "  make migrate-down - Revert one Alembic migration"

run:
	uv run uvicorn server.app.main:app --reload --host 0.0.0.0 --port 8001

test:
	uv run pytest -q

migrate-up:
	uv run alembic upgrade head

migrate-down:
	uv run alembic downgrade -1
