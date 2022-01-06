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
	rm -f docs/*.md
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

prepare_data:
	twitchatds data \
		--csv-path /media/data/twitchat-data-train \
		--channel zerator squeezie samueletienne ponce mistermv jeanmassietaccropolis domingo blitzstream antoinedaniellive \
		--out-file /media/data/Projets/twitchat-ds/data/raw/zerator_squeezie_samueletienne_ponce_mistermv_jeanmassietaccropolis_domingo_blitzstream_antoinedaniellive.pkl

train_tokenizer:
	twitchatds train_tokenizer \
		--in-file /media/data/Projets/twitchat-ds/data/raw/zerator_squeezie_samueletienne_ponce_mistermv_jeanmassietaccropolis_domingo_blitzstream_antoinedaniellive.pkl \
		--out-file /media/data/Projets/twitchat-ds/models/tokenizers/all_streamers.json \
		--vocab-size 16000 \
		--max-length 500

create_electra_data:
	twitchatds electra_data \
		--in-file /media/data/Projets/twitchat-ds/data/raw/zerator_squeezie_samueletienne_ponce_mistermv_jeanmassietaccropolis_domingo_blitzstream_antoinedaniellive.pkl \
		--out-directory data/raw/electra \
		-l 170 \
		--time-window-freq 60s \

convert_convbert:
	python -m transformers.models.convbert.convert_convbert_original_tf1_checkpoint_to_pytorch_and_tf2 \
		--tf_checkpoint_path /mnt/twitchat/models/convbert-small-bckp/ \
		--convbert_config_file /media/data/Projets/twitchat-ds/twitchatds/convbert_small_config.json \
		--pytorch_dump_path /mnt/twitchat/models/convbert-small-hf/

# old
train_mlm:
	twitchatds train_mlm \
		--model_name_or_path /mnt/twitchat/models/convbert-small-hf \
		--data_file data/raw/jeanmassietaccropolis.pkl \
		--tokenizer_file models/tokenizers/all_streamers.json \
		--output_dir models/mlm/convbert-small \
		--lr_scheduler_type constant \
		--learning_rate 5e-4 \
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
		--do_train \
		--resume_from_checkpoint 1

train_mlm_gpu:
	twitchatds train_mlm \
		--model_name_or_path ~/cloudfiles/code/Users/assets/models/convbert-small-hf/ \
		--data_file ~/cloudfiles/code/Users/assets/data/raw/zerator_squeezie_samueletienne_ponce_mistermv_jeanmassietaccropolis_domingo_blitzstream_antoinedaniellive.pkl \
		--tokenizer ~/cloudfiles/code/Users/assets/models/tokenizers/all_streamers.json \
		--output_dir ~/cloudfiles/code/Users/assets/models/convbert-small-hf-mlm/ \
		--lr_scheduler_type constant \
		--learning_rate 5e-4 \
		--num_train_epochs 1 \
		--per_device_train_batch_size 16 \
		--evaluation_strategy steps \
		--eval_steps 2000 \
		--per_gpu_eval_batch_size 16 \
		--logging_strategy steps \
		--logging_steps 20 \
		--save_strategy steps \
		--save_steps 2000 \
		--save_total_limit 10 \
		--log_level debug \
		--time_window_freq 10s \
		--max_length 512 \
		--do_train

train_simcse:
	twitchatds train_simcse \
		--in-file /mnt/twitchat/data/raw/zerator_squeezie_samueletienne_ponce_mistermv_jeanmassietaccropolis_domingo_blitzstream_antoinedaniellive.pkl \
		--model-name-or-path /mnt/twitchat/models/convbert-small-hf \
		--out-directory /mnt/twitchat/models/convbert-small-simcse \
		--batch-size 128 \
		--n-sample 1000 \
		--num-train-epochs 1

train_simcse_gpu:
	twitchatds train_simcse \
		--in-file ~/cloudfiles/code/Users/assets/data/raw/zerator_squeezie_samueletienne_ponce_mistermv_jeanmassietaccropolis_domingo_blitzstream_antoinedaniellive.pkl \
		--model-name-or-path ~/cloudfiles/code/Users/assets/models/convbert-small-hf \
		--out-directory ~/cloudfiles/code/Users/assets/models/convbert-small-simcse \
		--batch-size 32 \
		--n-sample 500000 \
		--num-train-epochs 3

prepare_data_valid:
	twitchatds data \
		--csv-path /mnt/twitchat/twitchat-data-valid \
		--channel zerator squeezie samueletienne ponce mistermv jeanmassiet domingo blitzstream antoinedaniellive doigby etoiles \
		--out-file /mnt/twitchat/data/raw/valid_zerator_squeezie_samueletienne_ponce_mistermv_jeanmassiet_domingo_blitzstream_antoinedaniellive_doigby_etoiles.pkl
