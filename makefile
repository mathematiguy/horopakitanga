# This Makefile automates routine tasks for this Singularity-based project.
IMAGE ?= container.sif
RUN ?= singularity exec $(FLAGS) $(IMAGE)
SINGULARITY_ARGS ?=
DVC_CACHE_DIR ?= $(shell dvc cache dir)
FLAGS ?= --nv -B $$(pwd):/code --pwd /code -B $(DVC_CACHE_DIR) -B ataarangi:/pkg/ataarangi --env MPLCONFIGDIR=/code/.matplotlib --env HF_HOME=/code/.cache
VENV_PATH ?= venv
PYTHON ?= /opt/local/bin/python3.11

include cluster/makefile

.PHONY: report show_logs trigger scratch archive repro predict start jupyter container push shell

report:
	$(RUN) bash -c "cd report && latexmk acl2023.tex -pdf"

optimise:
	$(RUN) $(PYTHON) ataarangi/optimise.py --train_path data/train_set.csv --dev_path data/dev_set.csv --model_folder models

train:
	$(RUN) $(PYTHON) ataarangi/train.py --batch_size 32 --lr 0.0001 --num_layers 10 --embed_size 128 --hidden_size 512 --epochs 100 --architecture lstm --train_path data/train_set.csv --dev_path data/dev_set.csv

label:
	$(RUN) $(PYTHON) ataarangi/labelling.py

run:
	$(RUN) bash run.sh

jupyter:
	sudo singularity exec $(FLAGS) sandbox.sif jupyter lab \
		--ip=0.0.0.0 \
		--no-browser \
		--port 8888 \
    --allow-root

clean:
	rm -f report/*.blg report/*.fls report/*.out report/*.log report/*.fdb_latexmk report/*.aux report/*.pdf report/*.bbl report/*.toc

init:
	dvc init && mkdir -p .dvc/cache

# Builds a Singularity container from the Singularity definition file.
# Note: This command requires sudo privileges.
container: $(IMAGE)
$(IMAGE): Singularity requirements.txt
	sudo singularity build --force $(IMAGE) $(SINGULARITY_ARGS) Singularity

# Starts a shell within the Singularity container, with the virtual environment activated.
shell:
	singularity shell $(FLAGS) $(IMAGE) $(SINGULARITY_ARGS) bash

sandbox: sandbox.sif
sandbox.sif: $(IMAGE)
	sudo singularity build --force --sandbox sandbox.sif $(IMAGE)

sandbox-shell: sandbox.sif
	sudo singularity shell --writable sandbox.sif
