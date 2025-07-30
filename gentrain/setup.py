from setuptools import setup

setup(
    name="gentrain",
    version="0.1.2",
    description="A example Python package",
    author="Ben Kräling",
    author_email="ben.kraeling@study.hs-duesseldorf.de",
    packages=["gentrain"],
    install_requires=["biopython", "networkx", "matplotlib", "pandas", "numpy"],
    package_data={"gentrain": ["reference.fasta"]},
    include_package_data=True,
)
