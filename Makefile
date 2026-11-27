# `uv run` installs anything missing from uv.lock before running.
PYTHON ?= uv run python

.PHONY: setup techniques fetch dataset eda holdout test evaluate random-baseline train-retriever predict-retriever retriever crossfit mine train-verifier predict-verifier verifier curves llm-baseline smoke

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

# Cross-fitting: FOLDS retrievers, each trained without one fold of train documents, rank the
# sentences of the fold they did not see. Gives the verifier train candidates like dev/test ones.
FOLDS ?= 5
crossfit:
	for f in $$(seq 0 $$(($(FOLDS) - 1))); do \
		$(PYTHON) -m src.retriever train --run $(RUN) --seed $(SEED) --folds $(FOLDS) --fold $$f $(ARGS) || exit 1; \
	done
	$(PYTHON) -m src.retriever crossfit --run $(RUN) --seed $(SEED) --folds $(FOLDS)

VERIFIER_DIR = runs/$(RUN)/verifier/seed$(SEED)

# Verifier training pairs from the retriever's mistakes on train (needs crossfit first).
mine:
	$(PYTHON) -m src.mine_negatives --run $(RUN) --seed $(SEED) $(MINE_make ARGS)

train-verifier:
	$(PYTHON) -m src.verifier train --run $(RUN) --seed $(SEED) $(ARGS)

# Two-stage pipeline: retriever top-k -> verifier. Tunes k and theta_v on dev, scores dev and test.
predict-verifier:
	$(PYTHON) -m src.verifier predict --run $(RUN) --seed $(SEED)
	$(PYTHON) -m evaluation.evaluate --pred $(VERIFIER_DIR)/dev.jsonl --split dev $(WITHHELD_FLAG) --out $(VERIFIER_DIR)/metrics_dev.json > /dev/null
	$(PYTHON) -m evaluation.evaluate --pred $(VERIFIER_DIR)/test.jsonl --split test $(WITHHELD_FLAG) --out $(VERIFIER_DIR)/metrics_test.json

verifier: mine train-verifier predict-verifier

# Training curves and a trained-properly verdict per seed, from every history.json of RUN.
# Writes runs/<RUN>/<model>/curves.png and curves.json.
curves:
	$(PYTHON) -m evaluation.curves --run $(RUN)

# Reference baseline: a local LLM through Ollama, zero-shot (needs `ollama serve` and the model pulled).
# ~5 hours for dev + test; rerun to resume. One set of predictions serves both run types:
# metrics_test_withheld.json has the seen/withheld groups for hypothesis 3.
LLM_MODEL ?= llama3.1:8b
LLM_DIR = runs/llm/$(subst /,-,$(subst :,-,$(LLM_MODEL)))
llm-baseline:
	$(PYTHON) -m src.llm_baseline predict --model $(LLM_MODEL) $(ARGS)
	$(PYTHON) -m evaluation.evaluate --pred $(LLM_DIR)/dev.jsonl --split dev --out $(LLM_DIR)/metrics_dev.json > /dev/null
	$(PYTHON) -m evaluation.evaluate --pred $(LLM_DIR)/test_untuned.jsonl --split test --out $(LLM_DIR)/metrics_test_untuned.json > /dev/null
	$(PYTHON) -m evaluation.evaluate --pred $(LLM_DIR)/test.jsonl --split test --withheld-run --out $(LLM_DIR)/metrics_test_withheld.json > /dev/null
	$(PYTHON) -m evaluation.evaluate --pred $(LLM_DIR)/test.jsonl --split test --out $(LLM_DIR)/metrics_test.json

# Quick check that the whole chain runs (a few hundred sentences, 1 epoch, a few minutes).
# Writes to runs/smoke/; the scores mean nothing.
smoke:
	$(PYTHON) -m src.retriever train --smoke
	$(PYTHON) -m src.retriever predict --smoke
	$(PYTHON) -m evaluation.evaluate --pred runs/smoke/retriever/test.jsonl --split test > /dev/null
	$(PYTHON) -m src.retriever train --smoke --folds 2 --fold 0
	$(PYTHON) -m src.retriever train --smoke --folds 2 --fold 1
	$(PYTHON) -m src.retriever crossfit --smoke --folds 2
	$(PYTHON) -m src.mine_negatives --smoke
	$(PYTHON) -m src.verifier train --smoke
	$(PYTHON) -m src.verifier predict --smoke
	$(PYTHON) -m evaluation.curves --smoke
