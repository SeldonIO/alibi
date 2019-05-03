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
	# sphinx-apidoc -o doc/source/api alibi '**/*test*' -M
	$(MAKE) -C doc html

.PHONY: clean_docs
clean_docs:
	$(MAKE) -C doc clean
	rm -r doc/source/api

.PHONY: build_pypi
build_pypi:
	python setup.py sdist bdist_wheel

.PHONY: push_pypi_test
push_pypi_test:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: push_pypi
push_pypi:
	twine upload dist/*
