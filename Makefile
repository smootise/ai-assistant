.PHONY: lint test check

lint:
	flake8 src --max-line-length=100

test:
	pytest --maxfail=1 --disable-warnings -q

check: lint test
