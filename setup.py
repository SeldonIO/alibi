from setuptools import find_packages, setup


def readme():
    with open('README.md') as f:
        return f.read()


# read version file
exec(open('alibi/version.py').read())

extras_require = {
    'examples': ['seaborn', 'Keras', 'xgboost']
}

setup(name='alibi',
      author='Seldon Technologies Ltd.',
      author_email='hello@seldon.io',
      version=__version__,  # type: ignore # noqa F821
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
          'attrs',
          'beautifulsoup4',
          'numpy',
          'Pillow',
          'pandas',
          'prettyprinter',
          'requests',
          'scikit-learn',
          'spacy',
          'scikit-image',
          'tensorflow<2.0',
          'shap',
          'scipy'
      ],
      tests_require=[
          'pytest',
          'pytest-cov',
          'pytest-xdist',
          'pytest-lazy-fixture',
      ],
      extras_require=extras_require,
      test_suite='tests',
      zip_safe=False)
