from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mujoco==3.5.0",
    "warp-lang==1.12",
    "mjlab==1.2.0",
    "mujoco-warp==3.5.0",
    "scipy==1.17.1",
    "onnxruntime==1.27.0",
    "pygame==2.6.1",
    "numpy",
    "scipy",
    "torchrunx",
    "tyro",
]

setup(
    name="go2_mjlab",
    version="0.0.1",
    packages=find_packages(include=("src", "src.*")),
    install_requires=INSTALL_REQUIRES
)
