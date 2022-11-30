import os
from setuptools import setup, find_packages


with open(os.path.join(os.path.dirname(__file__), "requirements.txt"), "r", encoding="utf-8") as req:
    requirements = req.read().strip().splitlines()
with open(os.path.join(os.path.dirname(__file__), "README.md"), "r", encoding="utf-8") as readme:
    readme_text = readme.read()
setup(
    name="auglm-chatbots",
    version="0.0.9",
    packages=find_packages(include=["aulm_chatbots", "aulm_chatbots.*"]),
    install_requires=requirements,
    author="Alexander Pozharskii",
    author_email="gaussmake@gmail.com",
    url="https://github.com/alex4321/gpt3-retrieving-bot",
    data_files=[
        ('', ['README.md', 'requirements.txt', 'aulm_chatbots/prompt.txt']),
    ],
    include_package_data=True,
    long_description=readme_text,
    long_description_content_type='text/markdown',
)