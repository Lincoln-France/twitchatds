.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT


help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with flake8
	flake8 twitchatds tests

test: ## run tests quickly with the default Python
	python setup.py test

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source twitchatds setup.py test
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/twitchatds.rst
	rm -f docs/modules.rst
	rm docs/*.md
	cp *.md docs
	sphinx-apidoc -o docs/ twitchatds
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install

install-dev: 
	pip install -e .

start-notebook:
	echo "Starting jupyter notebook on port 8888 with token \"lincoln\"."
	$(CONDA_EXE) run -n twitchat-ds python -m jupyter notebook --NotebookApp.token=lincoln --port=8888 --no-browser --ip=0.0.0.0

setup-conda-env:
	@if conda env list | grep -q "^twitchat-ds"; then\
		echo "/!\ Failed to create conda environnement. An environnement with name twitchat-ds already exists.";\
	else\
		$(CONDA_EXE) env create -f ./environment.yml > /dev/null;\
		$(CONDA_EXE) run -n twitchat-ds python -m ipykernel install --user --name=twitchat-ds;\
	fi

#
# PROJECT RECIPES
#

# local
export_env:
	export TRANSFORMERS_CACHE=/media/data/transformers

train_mlm:
	twitchatds train_mlm \
		--data_file data/raw/jeanmassietaccropolis.pkl \
		--tokenizer_file models/tokenizers/jeanmassietaccropolis.json \
		--output_dir models/mobilebert/mlm/ \
		--num_train_epochs 2 \
		--per_device_train_batch_size 16 \
		--evaluation_strategy steps \
		--eval_steps 60 \
		--logging_strategy steps \
		--logging_steps 20 \
		--save_strategy steps \
		--save_steps 100 \
		--save_total_limit 10 \
		--log_level debug \
		--time_window_freq 10s \
		--max_length 512\
		--do_train 

train_mlm_gpu:
	twitchatds train_mlm \
		--data_file ~/cloudfiles/code/Users/assets/data/raw/jeanmassietaccropolis.pkl \
		--tokenizer ~/cloudfiles/code/Users/assets/models/tokenizers/jeanmassietaccropolis.json \
		--output ~/cloudfiles/code/Users/assets/models/mobilebert/mlm \
		--resume_from_checkpoint ~/cloudfiles/code/Users/assets/models/mobilebert/mlm \
		--num_train_epochs 20 \
		--per_device_train_batch_size 16 \
		--evaluation_strategy steps \
		--eval_steps 500 \
		--logging_strategy steps \
		--logging_steps 20 \
		--save_strategy steps \
		--save_steps 500 \
		--save_total_limit 10 \
		--log_level debug \
		--time_window_freq 10s \
		--max_length 512 \
		--do_train 
