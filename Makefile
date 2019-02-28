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
