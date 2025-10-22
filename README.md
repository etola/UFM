
<div align="center">
<h1>UFM: A Simple Path towards Unified Dense Correspondence with Flow</h1>
<a href="https://uniflowmatch.github.io/assets/UFM.pdf"><img src="https://img.shields.io/badge/paper-blue" alt="Paper"></a>
<a href="https://arxiv.org/abs/2506.09278"><img src="https://img.shields.io/badge/arXiv-2506.09278-b31b1b" alt="arXiv"></a>
<a href="https://uniflowmatch.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href='https://huggingface.co/spaces/infinity1096/UFM'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>


**Carnegie Mellon University**

[Yuchen Zhang](https://infinity1096.github.io/), [Nikhil Keetha](https://nik-v9.github.io/), [Chenwei Lyu](https://www.linkedin.com/in/chenwei-lyu/), [Bhuvan Jhamb](https://www.linkedin.com/in/bhuvanjhamb/), [Yutian Chen](https://www.yutianchen.blog/about/), [Yuheng Qiu](https://haleqiu.github.io), [Jay Karhade](https://jaykarhade.github.io/), [Shreyas Jha](https://www.linkedin.com/in/shreyasjha/), [Yaoyu Hu](http://www.huyaoyu.com/), [Deva Ramanan](https://www.cs.cmu.edu/~deva/), [Sebastian Scherer](https://theairlab.org/team/sebastian/), [Wenshan Wang](http://www.wangwenshan.com/)
</div>

<p align="center">
    <img src="assets/teaser.jpg" alt="example" width=80%>
    <br>
    <em>UFM unifies the tasks of Optical Flow Estimation and Wide Baseline Matching and provides accurate dense correspondences for in-the-wild images at significantly fast inference speeds.</em>
</p>

## Welcome to the Benchmark Branch
If you reach here, you are interested in verifying the claims made in the UFM paper, thank you for the interest and for contributing to reproducible research! 

This branch contains the exact configurations, pretrained checkpoints, and evaluation scripts used in our experiments. It is intended to enable others to reproduce benchmark results, analyze ablations, and compare against UFM under standardized settings.

## Updates
- [2025/10/20] Benchmark & data script for primary results.
- [2025/10/08] Released 980 resolution models.
- [2025/06/10] Initial release of model checkpoint and inference code.

## What's Included

- **Pre-trained UFM models** in 560×420 and 980×644 resolutions (base and refine variants)
- **Benchmark datasets** with automated download scripts for Sintel, KITTI, ETH3D, DTU, and TA-WB
- **Preprocessed data packages** for ETH3D and DTU to skip hours of processing
- **Standardized evaluation pipeline** supporting all UFM model/dataset combinations
- **Reproducible experimental configurations** matching the paper results

## Getting Started

### 1. Installing the Benchmark Package
The benchmark branch hosts all benchmark code as a separate python package at `/home/inf/UFM/benchmarks`. 

```bash
cd benchmarks
pip install -e .

# re-install the base package
cd ..
pip install -e .
```

### 2. Evaluation Data Prepration
For faster verification of results, we provide both preprocessed benchmark datasets and processing scripts from the raw data. All data download scripts are located at `preprocess_datasets/download_scripts`. Please run the ones corresponding to `Sintel`, `KITTI`, `ETH3D`, `DTU`, `TA-WB`, and the metadata download script (required for `ETH3D` and `DTU` benchmarks). All script should be used as:

```bash
bash download_script.sh /path/to/data_root
```
Where `data_root` is a shared folder to host all benchmark data. 

#### Preprocessed Data
For quick verification of results and avoid hours of downloading and preprocessing, we directly provide processed data for `ETH3D` and `DTU`. To use them, run 

```bash
bash preprocess_datasets/download_scripts/download_eth3d_processed.sh /path/to/data_root
bash preprocess_datasets/download_scripts/download_dtu_processed.sh /path/to/data_root
```

#### Raw Data
We also provide complete data processing script to start from the raw data at `preprocess_datasets`.



### 3. Checkpoints Downloading
All `UFM` checkpoints will be automatically downloaded from HuggingFace.

### 4. Run evaluation
Run evaluation by 
```bash
python scripts/benchmark.py \ 
    --solution_name [soln_name] \
    --benchmark_name [bench_name] \
    --dataset_root [data_root] \
    --output_root [output_root]
```

Where `[data_root]` is the root folder in data processing, and `[soln_name]` and `[bench_name]` can be any combination of:

Solution name: `ufm_base_560`, `ufm_refine_560`, `ufm_base_980`, `ufm_refine_980`.

Benchmark name: `sintel_clean`, `sintel_final`, `kitti`, `eth3d_dense`, `dtu_dense`, `ta_wb_dense`.

The results will be printed to the screen and also recorded as a text file in `[output_root]`.

## Notes
- The branch is frozen for reproducibility - please open issues instead of direct commits for changes or extensions.
- If you wish to integrate UFM into your own pipeline, please refer to the main branch.
- For any discrepancies between your reproduced results and the paper, please open a issue.
- As explained in the paper, all `980` resolution checkpoints finetune only on high resolution data, which happens to be all from Optical Flow, as their last step. Thus they have degraded performance for wide-baseline.
  

### Reproducibility Notes
- `ufm_980_refine` is a updated checkpoint that is slightly different from what's reported in the paper. It have slightly better performance on optical flow.


## License

This code is licensed under a fully open-source [BSD-3-Clause license](LICENSE). The pre-trained UFM model checkpoints inherit the licenses of the underlying training datasets and as result, may not be used for commercial purposes (CC BY-NC-SA 4.0). Please refer to the respective training dataset licenses for more details.

Based on community interest, we can look into releasing an Apache 2.0 licensed version of the model in the future. Please upvote the issue [here](https://github.com/UniFlowMatch/UFM/issues/1#issue-3135416718) if you would like to see this happen.

## Acknowledgements

We thank the folowing projects for their open-source code: [DUSt3R](https://github.com/naver/dust3r), [MASt3R](https://github.com/naver/mast3r), [RoMA](https://github.com/Parskatt/RoMa), and [DINOv2](https://github.com/facebookresearch/dinov2).

## Citation
If you find our repository useful, please consider giving it a star ⭐ and citing our paper in your work:

```bibtex
@inproceedings{zhang2025ufm,
 title={UFM: A Simple Path towards Unified Dense Correspondence with Flow},
 author={Zhang, Yuchen and Keetha, Nikhil and Lyu, Chenwei and Jhamb, Bhuvan and Chen, Yutian and Qiu, Yuheng and Karhade, Jay and Jha, Shreyas and Hu, Yaoyu and Ramanan, Deva and Scherer, Sebastian and Wang, Wenshan},
 booktitle={arXiV},
 year={2025}
}
```
