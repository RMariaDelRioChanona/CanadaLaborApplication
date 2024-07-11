from os.path import abspath, dirname, join

from setuptools import find_packages, setup

this_dir = abspath(dirname(__file__))

with open(join(this_dir, "README.md"), encoding="utf-8") as file:
    long_description = file.read()

with open(join(this_dir, "requirements.txt")) as f:
    requirements = f.read().split("\n")

setup(
    name="labour-abm",
    version="0.1.1",
    description="Labour ABM",
    url="",
    long_description_content_type="text/markdown",
    long_description=long_description,
    author="María del Río Chanona",
    author_email="",
    license="CCBY4",
    install_requires=requirements,
    packages=find_packages(exclude=["docs"]),
    package_data={"labour_abm_canada": ["data/*", "*.yaml"]},
    include_package_data=True,
    entry_points={"console_scripts": ["labour_abm_canada=labour_abm_canada.__main__:main"]},
)
