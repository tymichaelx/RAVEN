# `uv run` installs anything missing from uv.lock before running.
PYTHON ?= uv run python

.PHONY: setup techniques fetch dataset eda holdout test evaluate random-baseline

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
