
setup-environment-ldm:
	conda env create -f latent-diffusion/environment.yml && \
	conda activate ldm && \
	pip install transformers==4.19.2 scann kornia==0.6.4 torchmetrics==0.6.0 && \
	pip install git+https://github.com/arogozhnikov/einops.git && \
	pip install --upgrade einops && \
	python3 -m pip install -U pip setuptools wheel && \
	python3 -m pip install -r requirements.txt && \
	python3 -m pip install -e .



autoencoder:
	python3 main.py -b ./latent-diffusion/models/first_stage_models/tiny.yaml --train --devices gpus --auto_select_gpus --accelerator gpu --gpus "0," --accumulate_grad_batches 5

latent_diffusion:
	python3 main.py -b ./latent-diffusion/models/ldm/tiny.yaml --train --devices gpus --auto_select_gpus --accelerator gpu --gpus "0," --accumulate_grad_batches 2 --auto_scale_batch_size

# other interresting flags:

# for optimal training
# --auto_scale_batch_size --auto_lr_find

# for debugging
# --overfit_batches 2 --check_val_every_n_epoch 2000
# python3 main.py -b ./models/ldm/tiny_uncond/config.yaml --train --devices gpus --auto_select_gpus --accelerator gpu --gpus "0," --accumulate_grad_batches 3 --overfit_batches 2 --auto_lr_find --check_val_every_n_epoch 2000