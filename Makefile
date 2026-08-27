.PHONY: smoke run analyze device

smoke:
	python tests/smoke_test.py
	python tests/driver_smoke_test.py

device:
	python check_device.py

run:
	python run_experiments.py --out results/paper

analyze:
	python analyze_results.py results/paper
