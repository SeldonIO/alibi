.PHONY: install
install:
	pip install -e .

.PHONY: test
test:
	python setup.py test

.PHONY: lint
lint:
	flake8 .

.PHONY: mypy
mypy:
	mypy .

.PHONY: build_docs
build_docs:
	sphinx-apidoc -o doc/source/api alibi '**/*test*' -M
	$(MAKE) -C doc html
