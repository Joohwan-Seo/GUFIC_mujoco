from setuptools import setup, find_packages

setup(
    name="gufic_env",          # the package name you'll import
    version="0.1.0",
    packages=find_packages(),  # finds gufic_env and subpackages
    install_requires=[
        # put required libs here if you want, e.g.
        # "mujoco",
        # "numpy",
    ],
)