PYTHON ?= python
PKG = image_tagger
SRC = $(PKG) tests

.PHONY: install-dev fmt lint style test coverage check

install-dev:
	$(PYTHON) -m pip install -e .[dev]

fmt:
	$(PYTHON) -m ruff format $(SRC)

lint:
	$(PYTHON) -m ruff check $(SRC)

style:
	$(PYTHON) -m pycodestyle $(SRC)

test:
	$(PYTHON) -m pytest

coverage:
	$(PYTHON) -m coverage run -m pytest && $(PYTHON) -m coverage report

check: lint style test
