from setuptools import setup, find_packages

setup(
    name="GeometryTools",
    version="0.2",
    packages=find_packages(),

    install_requires=[
        "numpy",
        "matplotlib",
        "scipy"
    ],

    author="Teddy Weisman",
    author_email="tjweisman@gmail.com",
    description="""Some basic tools for working with projective space, hyperbolic
    geometry, and group representations"""
)
