# `uv run` installs anything missing from uv.lock before running.
PYTHON ?= uv run python

.PHONY: setup techniques fetch dataset eda holdout test evaluate random-baseline train-retriever predict-retriever retriever smoke

setup:
	uv sync

# fetch_data.sh clones into ./raw relative to its working directory
fetch:
	cd data && bash fetch_data.sh

dataset: fetch
	$(PYTHON) data/build_dataset.py --raw data/raw --out data/extracted

techniques:
	$(PYTHON) src/techniques.py --stix data/raw/tram/data/attack/enterprise-attack.json

eda:
	$(PYTHON) src/eda.py

holdout:
	$(PYTHON) src/holdout.py

test:
	uv run pytest -q

# make evaluate PRED=path/to/predictions.jsonl [SPLIT=dev] [WITHHELD=1]
# WITHHELD=1 when the model was trained without the 10 holdout techniques (hypothesis 3).
SPLIT ?= test
evaluate:
	$(PYTHON) -m evaluation.evaluate --pred $(PRED) --split $(SPLIT) $(if $(WITHHELD),--withheld-run)

random-baseline:
	$(PYTHON) -m evaluation.evaluate --random-baseline --split $(SPLIT) --out evaluation/results/random_$(SPLIT).json

# ---------------------------------------------------------------- models
# RUN=full (hypotheses 1, 2) or RUN=withheld (hypothesis 3). Outputs go to runs/<RUN>/<model>/seed<SEED>/.
RUN ?= full
SEED ?= 0
RETRIEVER_DIR = runs/$(RUN)/retriever/seed$(SEED)
WITHHELD_FLAG = $(if $(filter withheld,$(RUN)),--withheld-run)

train-retriever:
	$(PYTHON) -m src.retriever train --run $(RUN) --seed $(SEED) $(ARGS)

# Tunes k and theta on dev, writes predictions and verifier candidates, scores dev and test.
predict-retriever:
	$(PYTHON) -m src.retriever predict --run $(RUN) --seed $(SEED)
	$(PYTHON) -m evaluation.evaluate --pred $(RETRIEVER_DIR)/dev.jsonl --split dev $(WITHHELD_FLAG) --out $(RETRIEVER_DIR)/metrics_dev.json > /dev/null
	$(PYTHON) -m evaluation.evaluate --pred $(RETRIEVER_DIR)/test.jsonl --split test $(WITHHELD_FLAG) --out $(RETRIEVER_DIR)/metrics_test.json

retriever: train-retriever predict-retriever

# Whole chain on a few hundred sentences, 1 epoch: catches bugs in minutes. Writes to runs/full/retriever/seed99/.
smoke:
	$(MAKE) retriever RUN=full SEED=99 ARGS="--limit 400 --epochs 1"
