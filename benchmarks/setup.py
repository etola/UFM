import os

from setuptools import setup

# Manually define the packages and their locations
packages = ["ufm_benchmarks"]
package_dir = {
    "ufm_benchmarks": "ufm_benchmarks",  # benchmarks package
}

setup(
    name="ufm_benchmarks",
    version="0.0.0",
    description="Benchmarks for UniFlowMatch Project",
    author="AirLab",
    license="BSD Clause-3",
    packages=packages,  # Directly specify the packages
    package_dir=package_dir,  # Define package directories
    include_package_data=True,
)
