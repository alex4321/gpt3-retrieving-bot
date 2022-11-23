import os
from setuptools import setup, find_packages


with open(os.path.join(os.path.dirname(__file__), "requirements.txt"), "r", encoding="utf-8") as req:
    requirements = req.read().strip().splitlines()
with open(os.path.join(os.path.dirname(__file__), "README.md"), "r", encoding="utf-8") as readme:
    readme_text = readme.read()
setup(
    name="aulm-chatbots",
    version="0.0.1",
    packages=find_packages(include=["chatbots", "chatbots.*"]),
    install_requires=requirements,
    author="Alexander Pozharskii",
    author_email="gaussmake@gmail.com",
    url="https://github.com/alex4321/gpt3-retrieving-bot",
    long_description=readme,
    long_description_content_type='text/markdown',
)