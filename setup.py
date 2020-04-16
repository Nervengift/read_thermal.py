import setuptools
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="flir-image-extractor-cli",
    version="1.0.1.post3",
    author="National Drones",
    author_email="development@nationaldrones.com",
    description="A cli-tool to get thermal information out of FLIR radiometric JPGs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/nationaldronesau/FlirImageExtractor",
    packages=["flir_image_extractor_cli"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "flir-image-extractor-cli=flir_image_extractor_cli.__main__:main"
        ]
    },
    install_requires=["numpy", "pillow", "matplotlib", "loguru"],
)
