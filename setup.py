import setuptools
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="flir-image-extractor-cli",
    version="1.0.1.post1",
    author="National Drones",
    author_email="development@nationaldrones.com",
    description="A cli-tool to get thermal information out of FLIR radiometric JPGs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/nationaldronesau/FlirImageExtractor",
    packages=["flir-image-extractor-cli"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "pillow", "matplotlib", "loguru"],
)
