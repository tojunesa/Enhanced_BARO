from os.path import join, dirname, abspath
from setuptools import setup

# parse requirements.txt to requirement list
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("requirements_rcd.lock") as f:
    rcd_requirements = f.read().splitlines()

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()
    
setup(
    name="RCAEval",
    version="0.4.3",
    packages=["RCAEval"],
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[],
    extras_require={"default": requirements, "rcd": rcd_requirements},
)
