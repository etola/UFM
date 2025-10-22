#!/bin/bash

# Change this to fit your PC and cluster setup
if [ -e "/ocean/projects" ]; then
    if [ -e "/local/datasets" ]; then
        machine=psc_local
    else
        machine=psc
    fi
else
    if [ -e "/data2" ]; then
        machine=cluster
    else
        machine=yuchen
    fi
fi

echo "Machine: ${machine}"

# change adjust effective_batch_size x accumulate_grad_batches = 48. Use 48 for 8 H100 gpus. 6 is just a demo.
python scripts/train.py \
    --config-name default \
    dataset=ufm_560_all \
    dataset/quantity_options/train=blendedmvs_100k \
    dataset/quantity_options/val=blendedmvs_full \
    loss=robust_epe_covisibility \
    machine=${machine} \
    training_scheme.effective_batch_size=6 \
    training_scheme.num_epochs=48 \
    training_scheme.warmup_epochs=4.8 \
    training_scheme.learning_rate.encoder=5e-6 \
    training_scheme.learning_rate.info_sharing=1e-4 \
    training_scheme.learning_rate.output_head=1e-4 \
    training_scheme.learning_rate.uncertainty_head=1e-4 \
    dataset.num_workers=12 \
    find_unused_parameters=True \
    visualizer=flow_covisible \
    training_scheme.validate_every_epochs=1 \
    training_scheme.accumulate_grad_batches=1 \
    training_scheme.symmetrize_inputs=True \
    hydra.run.dir=outputs/training/ufm_base_example \
    +wandb_tags=[example,blendedmvs] \
    +wandb_name=example_blendedmvs \
    resume_model=infinity1096/UFM-Base # load from huggingface, you can also load from .ckpt files
    # resume_all='"outputs/training/ufm_base_560/checkpoints_epoch/last.ckpt"' # resume from partial training