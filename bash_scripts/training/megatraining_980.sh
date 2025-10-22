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

python scripts/train.py \
    --config-name default \
    dataset=ufm_980_highres \
    loss=robust_epe_all_pixels \
    machine=${machine} \
    training_scheme.effective_batch_size=16 \
    training_scheme.num_epochs=15 \
    training_scheme.warmup_epochs=2 \
    training_scheme.learning_rate.encoder=5e-7 \
    training_scheme.learning_rate.info_sharing=1e-5 \
    training_scheme.learning_rate.output_head=1e-5 \
    training_scheme.learning_rate.uncertainty_head=0 \
    dataset.num_workers=12 \
    find_unused_parameters=True \
    visualizer=flow_covisible \
    training_scheme.validate_every_epochs=1 \
    training_scheme.accumulate_grad_batches=1 \
    training_scheme.symmetrize_inputs=True \
    hydra.run.dir=outputs/training/ufm_base_980 \
    +wandb_tags=[example,blendedmvs] \
    +wandb_name=example_blendedmvs \
    resume_model=infinity1096/UFM-Base # load from huggingface, you can also load from .ckpt files
    # resume_all='"outputs/training/ufm_base_980/checkpoints_epoch/last.ckpt"' # resume from partial training