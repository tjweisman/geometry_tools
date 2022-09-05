from setuptools import setup, find_packages
from os import path
from io import open

setup_dir = path.abspath(path.dirname(__file__))
with open(path.join(setup_dir, 'README.md'),
          encoding='utf-8') as readme_file:
    long_description = readme_file.read()

setup(
    name="geometry_tools",
    version="0.4",
    packages=find_packages(),
    include_package_data=True,

    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "setuptools-git"
    ],

    author="Teddy Weisman",
    author_email="tjweisman@gmail.com",
    license="MIT",
    url="https://github.com/tjweisman/geometry_tools",
    description="""Some tools for working with projective space, hyperbolic
    geometry, group representations, and finite state automata""",

    long_description=long_description,
    long_description_content_type="text/markdown"
)
