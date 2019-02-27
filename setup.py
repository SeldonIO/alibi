from setuptools import find_packages, setup
from alibi import __version__


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='alibi',
      author='Seldon Technologies Ltd.',
      author_email='hello@seldon.io',
      version=__version__,
      description='Algorithms for monitoring and explaining machine learning models',
      long_description=readme(),
      url='https://github.com/SeldonIO/alibi',
      license='Apache 2.0',
      packages=find_packages(),
      include_package_data=True,
      setup_requires=[
          'pytest-runner'
      ],
      install_requires=[
          'lime',
          'numpy',
          'pandas',
          'sklearn'
      ],
      tests_require=[
          'pytest',
          'pytest-cov'
      ],
      test_suite='tests',
      zip_safe=False)
