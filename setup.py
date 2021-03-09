import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ungol",
    version="0.1",
    author="Felix Hamann",
    author_email="felix@hamann.xyz",
    description="scalable document similarity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ramlimit.de/deepca/ungol/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
)
