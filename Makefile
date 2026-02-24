.PHONY: test smoke install

test:
	python -m pytest -q

smoke:
	python -c "import smaps; print('ok')"

install:
	pip install -e ".[dev]"
