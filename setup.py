from setuptools import find_packages, setup


def readme():
    with open('README.md') as f:
        return f.read()


# read version file
exec(open('alibi/version.py').read())

extras_require = {
    'examples': ['seaborn', 'Keras']
}

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
      python_requires='>3.5.1',
      setup_requires=[
          'pytest-runner'
      ],
      install_requires=[
          'beautifulsoup4',
          'numpy',
          'Pillow',
          'pandas',
          'requests',
          'scikit-learn',
          'spacy',
          'scikit-image',
          'tensorflow',
      ],
      tests_require=[
          'pytest',
          'pytest-cov'
      ],
      extras_require=extras_require,
      test_suite='tests',
      zip_safe=False)
