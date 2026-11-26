.PHONY: help install data train eval tables figures bench all clean

help:
	@echo "Targets:"
	@echo "  install  — pip install -r requirements.txt"
	@echo "  data     — generate dataset"
	@echo "  train    — train model"
	@echo "  eval     — evaluate"
	@echo "  tables   — regenerate Table 1 + 2"
	@echo "  figures  — regenerate Figure 1"
	@echo "  bench    — real-time benchmark"
	@echo "  all      — full pipeline"
	@echo "  clean    — remove artifacts/"

install:
	pip install -r requirements.txt

data:
	python data/generate_dataset.py --config config/default.yaml

train:
	python train.py --config config/default.yaml

eval:
	python evaluate.py --config config/default.yaml

tables:
	python experiments/table1_comparison.py
	python experiments/table2_taxonomy.py

figures:
	python experiments/figure1_architecture.py

bench:
	python experiments/real_time_benchmark.py

all:
	bash run_all_experiments.sh

clean:
	rm -rf artifacts/
