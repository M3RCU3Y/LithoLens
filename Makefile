.PHONY: install test prepare-force baseline demo app clone-references

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest

prepare-force:
	python scripts/prepare_force_data.py

baseline:
	python scripts/run_baseline.py --config configs/baseline.yaml

demo:
	python scripts/make_demo_outputs.py --config configs/baseline.yaml

app:
	streamlit run app/streamlit_app.py

clone-references:
	bash scripts/clone_references.sh
