from setuptools import setup, find_packages

install_requires = [line.strip() for line in open("requirements.txt")]

setup(
    name="digital_twin_learning",
    version="0.1.0",
    author="Lorenzo Piglia",
    author_email="lpiglia@ethz.ch",
    description="The digital_twin_learning package",
    packages=find_packages(),
    install_requires=install_requires,
)
