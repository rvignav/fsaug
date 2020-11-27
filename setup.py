from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='fsaug',
    version='0.1.0',
    description='Network training with feature space augmentation in Python',
    long_description_content_type="text/markdown",
    long_description=README,
    license='MIT',
    packages=find_packages(),
    author='Vignav Ramesh',
    author_email='rvignav@gmail.com',
    keywords=['Feature Space Augmentation', 'Feature Space', 'Data Augmentation', 'Python 3', 'CNN', 'Convolutional Neural Network', 'Neural Network', 'Training'],
    url='https://github.com/rvignav/fsaug',
    download_url='https://pypi.org/project/fsaug/'
)

install_requires = [
    'torch',
    'torchvision',
    'PIL',
    'numpy',
    'math',
    'random',
    'scipy',
    'numbers'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
