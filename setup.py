import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="flirimageextractor",
    version="0.0.2",
    author="Aidan Kinzett",
    author_email="a.kinzett@nationaldrones.com",
    description="A package to get thermal information out of FLIR radiometric JPGs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nationaldronesau/FlirImageExtractor",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)