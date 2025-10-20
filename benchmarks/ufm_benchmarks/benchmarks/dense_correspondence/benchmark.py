import os
from typing import Any, Dict

from ufm_benchmarks.base import UFMSolutionBase
from ufm_benchmarks.benchmarks.dense_correspondence.opticalflow_datasets import (
    KITTIDenseBenchmark,
    SintelDenseBenchmark,
)
from ufm_benchmarks.benchmarks.dense_correspondence.widebaseline_datasets import (
    DTUDenseBenchmark,
    ETH3DDenseBenchmark,
    TAWBDenseBenchmark,
)


def run_dense_benchmark(solution: UFMSolutionBase, benchmark_str: str, dataset_root: str, output_root: str):
    if benchmark_str == "sintel_clean":
        benchmark = SintelDenseBenchmark(
            solution=solution, output_root=output_root, sintel_root=os.path.join(dataset_root, "Sintel"), dstype="clean"
        )
    elif benchmark_str == "sintel_final":
        benchmark = SintelDenseBenchmark(
            solution=solution, output_root=output_root, sintel_root=os.path.join(dataset_root, "Sintel"), dstype="final"
        )
    elif benchmark_str == "kitti":
        benchmark = KITTIDenseBenchmark(
            solution=solution,
            output_root=output_root,
            kitti_root=os.path.join(dataset_root, "KITTI"),
        )
    elif benchmark_str == "eth3d_dense":
        benchmark = ETH3DDenseBenchmark(
            solution=solution,
            output_root=output_root,
            eth3d_root=os.path.join(dataset_root, "eth3d_processed"),
            eth3d_pairs_root=os.path.join(dataset_root, "anymap_data_pairs", "eth3d_valid_pairs.npz"),
        )
    elif benchmark_str == "dtu_dense":
        benchmark = DTUDenseBenchmark(
            solution=solution,
            output_root=output_root,
            dtu_root=os.path.join(dataset_root, "dtu_processed"),
            dtu_pairs_root=os.path.join(dataset_root, "anymap_data_pairs", "dtu_valid_pairs.npz"),
        )
    elif benchmark_str == "ta_wb_dense":
        benchmark = TAWBDenseBenchmark(
            solution=solution,
            output_root=output_root,
            dataset_root=os.path.join(dataset_root, "TA-WB"),
        )
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_str}")

    results = benchmark.run_all()
    return results
