from .ufm import UFMSolution


def get_solution(solution_name: str, ckpt_dir: str):
    if solution_name == "ufm_base_560":
        return UFMSolution(hf_repo="infinity1096/UFM-Base")
    elif solution_name == "ufm_refine_560":
        return UFMSolution(hf_repo="infinity1096/UFM-Refine")
    elif solution_name == "ufm_base_980":
        return UFMSolution(hf_repo="infinity1096/UFM-Base-980")
    elif solution_name == "ufm_refine_980":
        return UFMSolution(hf_repo="infinity1096/UFM-Refine-980")
    else:
        raise ValueError(f"Unknown solution name: {solution_name}")
