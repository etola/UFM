# Data Prepration

This script handles how to download and prepare all data required for UFM training.

## Metadata (Required for BlendedMVS, Scannet++)
Download the pairs metadata from theairlabcmu/UFM-Training-Metadata to be used with the preprocessing scripts.

## BlendedMVS
Download BlendedMVS from [here](https://github.com/YoYo000/BlendedMVS) and follow `/jet/home/yzhang25/UFM/preprocess_datasets/preprocess_blendedmvs.py`.

## MegaDepth
Follow DUSt3R's preprocessing script at [here](https://github.com/naver/dust3r/tree/main/datasets_preprocess).

## TA_WB Training Set
We are still uploading this dataset (2.0 TB) at https://huggingface.co/datasets/theairlabcmu/TA-WB. Please stay tuned.

## Scannet++ V2
Download Scannet++ V2 from [here](https://scannetpp.mlsg.cit.tum.de/scannetpp/) and follow `preprocess_datasets/preprocess_scannetpp.py`.

## Habitat CAD
Follow DUSt3R's preprocessing script at [here](https://github.com/naver/dust3r/tree/main/datasets_preprocess/habitat), but only process the `ReplicaCAD` data. Place the processed data as

```
data_root/
  habitat_processed/
    replica_cad_baked_lighting/
```

## Static Things
Follow DUSt3R's preprocessing script at [here](https://github.com/naver/dust3r/blob/main/datasets_preprocess/preprocess_staticthings3d.py). Place the processed data as `data_root/staticthings3d_processed/TRAIN`

## Kubrics 4D

Download the training set from https://gcd.cs.columbia.edu/gcd_kubric4d_train_dl.sh and the metadata files, place the files as:

```
kubric4d/
  kubric4d_raw/
    scn00000/
    scn00001/
    ...
  kubric4d_metadata/
```

## FlyingThings

```bash
bash preprocess_datasets/download_scripts/download_flyingthings.sh [data_root]
```

It will download raw data to `[data_root]/raw_data/FlyingThings`. (Download source: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

Then run the `preprocess_datasets/preprocess_flyingthings.py`

## FlyingChairs

```bash
bash preprocess_datasets/download_scripts/download_flyingchairs.sh [data_root]
```

It will directly setup `[data_root]/FlyingChairs`. (Download source: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

Download [the validation split](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/FlyingChairs_train_val.txt) and place it under `FlyingChairs/`.

## Spring
Download the training set from https://spring-benchmark.org and use `preprocess_datasets/preprocess_spring.py`.

## Monkaa

```bash
bash preprocess_datasets/download_scripts/download_monkaa.sh [data_root]
```

It will download raw data to [data_root]/raw_data/Monkaa. (Download source: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

Then run `preprocess_datasets/preprocess_monkaa.py`

## HD1K
The original data is hosted at [here](https://hci.iwr.uni-heidelberg.de/benchmarks). However, since the site is unreachable, we host a copy of [the original data](https://huggingface.co/datasets/theairlabcmu/HD1K_backup). 

process the data with `preprocess_datasets/preprocess_hd1k.py`.

## In case some dataset is hard to work with
You can always try to train with less data, by modifying `configs/dataset/quantity_options`.