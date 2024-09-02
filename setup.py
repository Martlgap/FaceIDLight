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
        "setuptools>=70.2.0,<74.2",
        "opencv-python>=4.5.1.48,<4.10.1.0",
        "numpy>=2.0.0,<2.1.0",
        "tqdm>=4.59,<4.67",
        "scikit-image>=0.17.2,<0.25.0",
        "matplotlib>=3.3.3,<3.10.0",
        "scipy>=1.4.1,<1.15.0",
        "scikit-learn>=0.24,<1.6",
        "tflite-runtime==2.14.0",
    ],
)
