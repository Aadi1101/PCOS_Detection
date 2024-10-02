"""
This is the setup configuration script for the 'PCOS Detection' package.

It handles the installation and packaging details of the project, including:
- Defining the package name, version, author information, and email.
- Specifying the required dependencies through the `requirements.txt` file.
- Automatically discovering all Python packages in the project.

The `get_requirements` function reads and processes dependencies from a requirements file,
removing any local directory install commands (e.g., '-e .').

To install the package, run:
    python setup.py install
"""
from typing import List
from setuptools import find_packages, setup

def get_requirements(file_path:str)->List[str]:
    """
    Reads the requirements file and returns a list of dependencies.
    
    Args:
        file_path (str): The path to the requirements.txt file.

    Returns:
        List[str]: A list of package dependencies.
    """
    requirements = []
    with open(file_path,encoding="utf-8") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name="PCOS Detection",
    version="0.0.1",
    author="Aaditya Komerwar",
    author_email="aadityakomerwar@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)
