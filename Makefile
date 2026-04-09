.PHONY: lint typecheck test smoke reproduce install dev

install:
	pip install -e .

dev:
	pip install -e ".[all]"

lint:
	ruff check src/ tests/

typecheck:
	pyright src/

test:
	pytest tests/unit/ -v

smoke:
	pytest tests/smoke/ -v -m "not gpu"

reproduce:
	bash scripts/reproduce.sh

fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/
