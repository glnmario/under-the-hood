import io
from setuptools import setup, find_packages

with io.open('./README.md', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='under-the-hood',
    version='0.1.0',
    url='https://github.com/Procope/under-the-hood',
    packages=find_packages(),
    license='GNU General Public License v3.0',
    long_description=readme,
)
