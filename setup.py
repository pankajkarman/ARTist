from setuptools import find_packages, setup

with open('requirements.txt') as f:
    install_req = [req.strip() for req in f.read().split('\n')]
install_req = [req for req in install_req if req and req[0] != '#']

with open("readme.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
     name='ArtViz',
     version='0.0.1',
     description='Post-processing and plotting ICON-ART output in python',
     long_description=long_description,
     long_description_content_type='text/markdown',
     url='https://github.com/pankajkarman/ArtViz',
     author='Pankaj Kumar',
     author_email='pankaj.kmr1990@gmail.com',
     license='MIT',
     packages=find_packages(),
     py_modules=['art'],
     install_requires=install_req,
     setup_requires=['setuptools'],
)
