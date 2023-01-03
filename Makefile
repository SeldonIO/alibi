.PHONY: install
install:
	pip install -e .

.PHONY: test
test:
	pytest alibi

.PHONY: lint
lint:
	flake8 alibi

.PHONY: mypy
mypy:
	mypy alibi

.PHONY: build_docs
build_docs:
	# readthedocs.org build command
	python -m sphinx -T -b html -d doc/_build/doctrees -D language=en doc/source  doc/_build/html

.PHONY: build_latex
build_latex: ## Build the documentation into a pdf
	# readthedocs.org build command
	# explicit cd here due to a bug in latexmk 4.41
	python -m sphinx -b latex -d doc/_build/doctrees -D language=en doc/source doc/_build/latex && \
	cd doc/_build/latex && \
	latexmk -pdf -f -dvi- -ps- -jobname=alibi -interaction=nonstopmode

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

.PHONY: licenses
licenses:
	# create a tox environment and pull in license information
	tox --recreate -e licenses
	cut -d, -f1,3 ./licenses/license_info.csv \
					> ./licenses/license_info.no_versions.csv

.PHONY: check_licenses
	# check if there has been a change in license information, used in CI
check_licenses:
	git --no-pager diff --exit-code ./licenses/license_info.no_versions.csv

.PHONY: repl
tox-env=default
repl:
	env COMMAND="python" tox -e $(tox-env)


