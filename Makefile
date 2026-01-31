.PHONY: help test lint plots wandb

help:
	@echo "Targets:"
	@echo "  test    run pytest"
	@echo "  plots   regenerate writeup/figures/*.png from runs/ + writeup/figures/data/"
	@echo "  wandb   pull training history into writeup/figures/data/"

test:
	pytest tests/ -q

plots:
	python scripts/make_plots.py

wandb:
	python scripts/fetch_wandb.py --out writeup/figures/data/
