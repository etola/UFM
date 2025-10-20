import argparse
import os

from ufm_benchmarks.benchmarks.dense_correspondence.benchmark import run_dense_benchmark
from ufm_benchmarks.solutions import get_solution

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dense benchmark")
    parser.add_argument("--benchmark_name", type=str, default="sintel_clean", help="Benchmark name")
    parser.add_argument("--solution_name", type=str, default="ufm_base_560", help="Solution name")
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/inf/UFM/outputs/ufm_benchmarks",
        help="Output root directory",
    )
    parser.add_argument("--dataset_root", type=str, default="/home/inf/UFM/data", help="Dataset root directory")
    parser.add_argument("--ckpt_root", type=str, default="/home/inf/UFM/checkpoints", help="Checkpoint root directory")

    args = parser.parse_args()

    print("Output root:", args.output_root)
    os.makedirs(args.output_root, exist_ok=True)

    solution = get_solution(args.solution_name, args.ckpt_root)

    result = run_dense_benchmark(
        solution=solution,
        benchmark_str=args.benchmark_name,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
    )
