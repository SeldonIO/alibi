from setuptools import find_packages, setup


def readme():
    with open('README.md') as f:
        return f.read()


# read version file
exec(open('alibi/version.py').read())

setup(name='alibi',
      author='Seldon Technologies Ltd.',
      author_email='hello@seldon.io',
      version=__version__,
      description='Algorithms for monitoring and explaining machine learning models',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/SeldonIO/alibi',
      license='Apache 2.0',
      packages=find_packages(),
      include_package_data=True,
      python_requires='>=3.5',
      setup_requires=[
          'pytest-runner'
      ],
      install_requires=[
          'beautifulsoup4',
          'keras',
          'numpy',
          'opencv-python',
          'pandas',
          'requests',
          'scikit-learn',
          'spacy',
          'scikit-image',
          'tensorflow',
          'seaborn'
      ],
      tests_require=[
          'pytest',
          'pytest-cov'
      ],
      test_suite='tests',
      zip_safe=False)
