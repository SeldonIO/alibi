from setuptools import find_packages, setup


def readme():
    with open('README.md') as f:
        return f.read()


# read version file
exec(open('alibi/version.py').read())

extras_require = {
    'examples': ['seaborn>=0.9.0', 'xgboost>=0.90'],
    'ray': ['ray>=0.8.7,<1.2'], # from requirements/dev.txt
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
      install_requires=[
          'numpy>=1.17.4,<1.21',
          'pandas>=0.23.4,<1.3',
          'scikit-learn>=0.21.2,<0.25',
          'spacy[lookups]>=2.0.18,<3.1',
          'scikit-image>=0.14.2,!=0.17.1,<0.19',  # https://github.com/SeldonIO/alibi/issues/215
          'requests>=2.21.0,<3.0.',
          'Pillow>=6.0.0,<8.2',
          'beautifulsoup4>=4.7.1,<5.0',
          'tensorflow>=2.0,<2.5',
          'attrs>=19.2.0,<21.0',
          'shap>=0.36.0,!=0.38.1,<0.39',  # https://github.com/SeldonIO/alibi/issues/333
          'scipy>=1.3.1,<1.7',
          'matplotlib>=3.1.2,<3.4',
          'typing-extensions>=3.7.2; python_version < "3.8"',
      ],
      extras_require=extras_require,
      test_suite='tests',
      zip_safe=False)
