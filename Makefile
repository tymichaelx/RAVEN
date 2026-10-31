PYTHON ?= .venv/bin/python

.PHONY: techniques

techniques:
	$(PYTHON) src/techniques.py --stix data/raw/tram/data/attack/enterprise-attack.json
