from os import path

from setuptools import find_packages, setup

import versioneer

cur_dir = path.abspath(path.dirname(__file__))

with open(path.join(cur_dir, "requirements.txt"), "r") as f:
    requirements = f.read().split()


setup(
    name="xopt",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    package_dir={"xopt": "xopt"},
    url="https://github.com/ChristopherMayes/xopt",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.7",
)
