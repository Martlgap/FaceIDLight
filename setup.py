from setuptools import setup, find_packages


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
    install_requires=[
        "setuptools>=51.0,<68.1",
        "opencv-python>=4.5.1.48,<4.8.1.0",
        "numpy>=1.19.5,<1.26.0",
        "tqdm>=4.59,<4.66",
        "scikit-image>=0.17.2,<0.22.0",
        "matplotlib>=3.3.3,<3.8.0",
        "scipy>=1.4.1,<1.12.0",
        "scikit-learn>=0.24,<1.4",
        "tflite-runtime>=2.7.0,<2.13.0",
    ],
)
