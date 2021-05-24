import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

print(setuptools.find_packages(include=["gma"]))

setuptools.setup(
    name="gma",
    version="0.0.1",
    author="Simon Wengeler",
    author_email="simon.wengeler@outlook.com",
    description="Packaged version of...",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/swengeler/GMA",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[""],
)
