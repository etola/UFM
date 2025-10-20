from uniflowmatch.datasets.base.flow_postprocessing import (
    PATHWAY_REQUIREMENTS,
    TEXT_QUANTITIES,
    apply_flow_postprocessing_and_merge_batch,
    collate_fn_with_delayed_flow_postprocessing,
    flow_occlusion_post_processing,
)

# benchmark datasets
from uniflowmatch.datasets.dtu import DTU
from uniflowmatch.datasets.eth3d import ETH3D
from uniflowmatch.datasets.tartanair_assembled import TartanairAssembled

__all__ = [
    "TartanairAssembled",
    "DTU",
    "ETH3D",
    "get_data_loader",
    "flow_occlusion_post_processing",
    "collate_fn_with_delayed_flow_postprocessing",
    "apply_flow_postprocessing_and_merge_batch",
]


# The following code is adopted from AnyMap
def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    persistent_workers=False,
    collate_fn=None,
):
    import torch

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    try:
        sampler = dataset.make_sampler(
            batch_size, shuffle=shuffle, world_size=world_size, rank=rank, drop_last=drop_last
        )
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    return data_loader
