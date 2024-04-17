from setuptools import setup, find_packages

setup(
    name="ataarangi",
    version="0.1",
    packages=find_packages(exclude=["tests*"]),
    description="A python package for experimenting with te ataarangi",
    url="https://github.com/mathematiguy/horopakitanga",
    author="Caleb Moses",
    author_email="calebjdmoses@gmail.com",
    include_package_data=True,
)
