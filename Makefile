PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: install lint test data features train evaluate predict serve drift up down clean pipeline

install:
	$(PIP) install -e ".[dev]"

lint:
	ruff check src tests

test:
	pytest

data:
	mlops-data

features:
	mlops-features

train:
	mlops-train

evaluate:
	mlops-evaluate

predict:
	mlops-predict

serve:
	mlops-serve

drift:
	mlops-drift

pipeline:
	dvc repro

up:
	docker compose up --build -d

down:
	docker compose down

clean:
	rm -rf data/raw data/features data/predictions artifacts/eval_metrics.json artifacts/taxi-fare-model.joblib