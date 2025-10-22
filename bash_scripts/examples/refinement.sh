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

# You are expected to see some evident refinement features after about 1500 steps.
python scripts/train.py \
    --config-name default \
    dataset=ufm_560_all \
    dataset/quantity_options/train=blendedmvs_100k \
    dataset/quantity_options/val=blendedmvs_full \
    loss=refinement_only \
    machine=${machine} \
    model=uniflowmatch_refinement \
    +model.encoder.encoder_kwargs.indices=[6,23] \
    training_scheme=refinement \
    training_scheme.effective_batch_size=6 \
    training_scheme.num_epochs=10 \
    training_scheme.warmup_epochs=1 \
    training_scheme.learning_rate.encoder=0 \
    training_scheme.learning_rate.info_sharing=0 \
    training_scheme.learning_rate.output_head=0 \
    training_scheme.learning_rate.uncertainty_head=0 \
    training_scheme.learning_rate.classification_head=1e-4 \
    training_scheme.learning_rate.unet_feature=1e-4 \
    dataset.num_workers=12 \
    find_unused_parameters=True \
    visualizer=flow_refinement \
    training_scheme.validate_every_epochs=1 \
    training_scheme.accumulate_grad_batches=1 \
    training_scheme.symmetrize_inputs=True \
    hydra.run.dir=outputs/training/ufm_refine_example \
    +wandb_tags=[example,blendedmvs,refine] \
    +wandb_name=example_refine_blendedmvs \
    resume_model=infinity1096/UFM-Base # initialize from the base model, train for refinement only.
    # resume_all='"outputs/training/ufm_refine_example/checkpoints_epoch/last.ckpt"' # resume from partial training