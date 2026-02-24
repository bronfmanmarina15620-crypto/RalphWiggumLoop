.PHONY: test lint typecheck smoke install

test:
	python -m pytest -q

lint:
	ruff check src/ tests/

typecheck:
	mypy src/

smoke:
	python -c "import smaps; print('ok')"

install:
	pip install -e ".[dev]"
