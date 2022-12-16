from setuptools import setup, find_packages

BASE_DEPENDENCIES = [
    "numpy",
    "scipy>=1.5.3",
    "jax>=0.2.23",
    "jaxlib>=0.1.69",
    "flax>=0.3.5",
    "optax>=0.0.2",
    "chex",
]

# MPI_DEPENDENCIES = ["mpi4py>=3.0.1, <4", "mpi4jax~=0.3.1"]

setup(
    name="continuousvmc",
    author="Matija Medvidovic",
    url="https://github.com/Matematija/continuous-vmc",
    author_email="matija.medvidovic@columbia.edu",
    license="Apache 2.0",
    description="ContinuousVMC : Continuous-Variable Variational Monte Carlo calculations",
    packages=find_packages(),
    install_requires=BASE_DEPENDENCIES,
    python_requires=">=3.7",
    # extras_require={"mpi": MPI_DEPENDENCIES},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
