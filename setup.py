import re
from setuptools import setup, find_packages


with open("requirements.txt", "r") as f:
    install_requires = f.readlines()

with open("chemdx_agent/__init__.py") as f:
    version = re.search(r"__version__ = [\'\"](?P<version>.+)[\'\"]", f.read()).group("version")


setup(
    name="chemdx_agent",
    version=version,
    description="Agent for ChemDX database",
    author="Yeonghun Kang",
    author_email="dudgns4675@gmail.com",
    packages=find_packages(),
    install_requires=install_requires,
    scripts=[],
    python_requires=">=3.9",
)
