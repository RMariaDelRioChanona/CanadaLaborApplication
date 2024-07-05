from os.path import abspath, dirname, join

from setuptools import find_packages, setup

this_dir = abspath(dirname(__file__))

with open(join(this_dir, "README.md"), encoding="utf-8") as file:
    long_description = file.read()

with open(join(this_dir, "requirements.txt")) as f:
    requirements = f.read().split("\n")

setup(
    name="labour-abm",
    version="0.1.0",
    description="Labour ABM",
    url="",
    long_description_content_type="text/markdown",
    long_description=long_description,
    author="María del Río Chanona",
    author_email="",
    license="CCBY4",
    install_requires=requirements,
    packages=find_packages(exclude=["docs"]),
    include_package_data=True,
)
