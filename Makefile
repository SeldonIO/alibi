.PHONY: install
install:
	pip install -e .

.PHONY: test
test:
	python setup.py test
	
