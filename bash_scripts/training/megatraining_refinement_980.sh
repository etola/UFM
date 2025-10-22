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
# change batch size to 32 for 8 gpus
python scripts/train.py \
    --config-name default \
    dataset=ufm_980_highres \
    loss=robust_epe_all_pixels_and_refinement \
    machine=${machine} \
    model=uniflowmatch_refinement \
    model.refinement_range=5 \
    +model.encoder.encoder_kwargs.indices=[6,23] \
    training_scheme=refinement \
    training_scheme.effective_batch_size=16 \
    training_scheme.num_epochs=20 \
    training_scheme.warmup_epochs=2 \
    training_scheme.learning_rate.encoder=5e-7 \
    training_scheme.learning_rate.info_sharing=1e-5 \
    training_scheme.learning_rate.output_head=1e-5 \
    training_scheme.learning_rate.uncertainty_head=1e-5 \
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
    +wandb_name=ufm_980_refine \
    resume_model=infinity1096/UFM-Base # initialize from the base model, train for refinement only.
    # resume_all='"outputs/training/ufm_refine_example/checkpoints_epoch/last.ckpt"' # resume from partial training
