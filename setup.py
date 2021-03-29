from setuptools import find_packages, setup


def readme():
    with open('README.md') as f:
        return f.read()


# read version file
exec(open('alibi/version.py').read())

extras_require = {
    'examples': ['seaborn>=0.9.0', 'xgboost>=0.90'],
    'ray': ['ray>=0.8.7, <2.0.0'],  # from requirements/dev.txt
    # shap is separated due to build issues, see https://github.com/slundberg/shap/pull/1802
    'shap': ['shap>=0.36.0, !=0.38.1, <0.40.0'],  # versioning: https://github.com/SeldonIO/alibi/issues/333
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
      python_requires='>=3.6',
      # lower bounds based on Debian Stable versions where available
      install_requires=[
          'numpy>=1.16.2, <2.0.0',
          'pandas>=0.23.3, <2.0.0',
          'scikit-learn>=0.20.2, <0.25.0',
          'spacy[lookups]>=2.0.0, <4.0.0',
          'scikit-image>=0.14.2, !=0.17.1, <0.19',  # https://github.com/SeldonIO/alibi/issues/215
          'requests>=2.21.0, <3.0.0',
          'Pillow>=5.4.1, <9.0',
          'tensorflow>=2.0.0, <2.5.0',
          'attrs>=19.2.0, <21.0.0',
          'scipy>=1.1.0, <2.0.0',
          'matplotlib>=3.0.0, <4.0.0',
          'typing-extensions>=3.7.2; python_version < "3.8"',  # https://github.com/SeldonIO/alibi/pull/248
      ],
      extras_require=extras_require,
      test_suite='tests',
      zip_safe=False)
