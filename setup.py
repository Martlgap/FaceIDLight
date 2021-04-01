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
        "setuptools~=51.0.0",
        "opencv-python~=4.5.1.48",
        "numpy~=1.19.5",
        "tqdm~=4.59.0",
        "scikit-image~=0.18.1",
        "matplotlib~=3.3.3",
        "scipy~=1.4.1",
        "scikit-learn~=0.24.0",
    ],
    include_package_data=True,
    zip_safe=False,
)
