from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    version="0.1",
    name="FaceIDLight",
    packages=find_packages(),
    url="https://github.com/Martlgap/FaceIDLight",
    author="Martin Knoche",
    author_email="martin.knoche@tum.de",
    license="MIT",
    description="A lightweight face-recognition toolbox.",
    long_description=open("README.md").read(),
    install_requires=requirements,
)
