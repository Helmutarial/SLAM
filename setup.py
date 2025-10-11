from setuptools import setup, find_packages

setup(
    name='collaborative_slam',
    version='0.1',
    description='Collaborative SLAM tools and scripts for OAK-D camera data processing',
    author='ArtAdmin',
    packages=find_packages(include=['collaborative_slam', 'collaborative_slam.*', 'utils', 'utils.*', 'views', 'views.*']),
    install_requires=[],  # Dependencies are managed in requirements.txt
    include_package_data=True,
)
