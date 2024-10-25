
setup-environment-ldm:
	conda env create -f latent-diffusion/environment.yml && \
	conda activate ldm && \
	pip install transformers==4.19.2 scann kornia==0.6.4 torchmetrics==0.6.0 && \
	pip install git+https://github.com/arogozhnikov/einops.git && \
	pip install --upgrade einops && \
	python3 -m pip install -U pip setuptools wheel && \
	python3 -m pip install -r requirements.txt && \
	python3 -m pip install -e .


# cd into latent-diffusion first
autoencoder:
	python3 main.py -b ./models/first_stage_models/shape_model_tiny.yaml --train --devices gpus --auto_select_gpus --accelerator gpu --gpus "0," --accumulate_grad_batches 1
	python3 main.py -b ./models/first_stage_models/height_model_small.yaml --train --devices gpus --auto_select_gpus --accelerator gpu --gpus "0," --accumulate_grad_batches 4

latent_diffusion:
	python3 main.py -b ./models/ldm/ldm-small.yaml --train --devices gpus --auto_select_gpus --accelerator gpu --gpus "0," --accumulate_grad_batches 3
	python3 main.py -b ./models/ldm/shape-ldm-tiny.yaml --train --devices gpus --auto_select_gpus --accelerator gpu --gpus "0," --accumulate_grad_batches 1
	python3 main.py -b ./models/ldm/height-ldm-small.yaml --train --devices gpus --auto_select_gpus --accelerator gpu --gpus "0," --accumulate_grad_batches 2

resume-ae:
	python3 main.py -b ./models/first_stage_models/height_model_small.yaml --train --devices gpus --auto_select_gpus --accelerator gpu --gpus "0," --accumulate_grad_batches 6 --resume_from_checkpoint ./logs/2024-10-24T13-17-49_height_model_small/checkpoints/epoch=000011.ckpt


# python3 main.py -b ./models/first_stage_models/shape_model_small.yaml --train --devices gpus --auto_select_gpus --accelerator gpu --gpus "0," --accumulate_grad_batches 2 --resume_from_checkpoint ./logs/2024-10-21T13-48-06_shape_model_small/checkpoints/last.ckpt


host-logs:
	tensorboard --logdir ./logs/

# other interresting flags:

# for optimal training
# --auto_scale_batch_size --auto_lr_find

# for debugging
# --overfit_batches 2 --check_val_every_n_epoch 2000
# python3 main.py -b ./models/ldm/tiny_uncond/config.yaml --train --devices gpus --auto_select_gpus --accelerator gpu --gpus "0," --accumulate_grad_batches 3 --overfit_batches 2 --auto_lr_find --check_val_every_n_epoch 2000



# python3 main.py -b ./models/first_stage_models/shape_model_small.yaml  --train --devices gpus --auto_select_gpus --accelerator gpu --gpus "0," --accumulate_grad_batches 3 --resume_from_checkpoint ./logs/2024-10-23T15-13-52_shape_model_small/checkpoints/last.ckpt


# python3 main.py -b ./models/ldm/height-ldm-small.yaml --train --devices gpus --auto_select_gpus --accelerator gpu --gpus "0," --accumulate_grad_batches 2 --resume_from_checkpoint ./logs/2024-10-24T16-53-08_height-ldm-small/checkpoints/last.ckpt 