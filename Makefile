# `uv run` installs anything missing from uv.lock before running.
PYTHON ?= uv run python

.PHONY: setup techniques fetch dataset eda holdout

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
