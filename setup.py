from setuptools import find_packages, setup


def readme():
    with open('README.md') as f:
        return f.read()


# read version file
exec(open('alibi/version.py').read())

extras_require = {
    'examples': ['seaborn', 'xgboost'],
    'ray': ['ray'],
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
          'attrs',
          'beautifulsoup4',
          'matplotlib',
          'numpy',
          'Pillow',
          'pandas',
          'requests',
          'scikit-learn',
          'spacy[lookups]',
          'scikit-image!=0.17.1',  # https://github.com/SeldonIO/alibi/issues/215
          'tensorflow>=2.0',
          'shap>=0.36,!=0.38.1',  # https://github.com/SeldonIO/alibi/issues/333
          'scipy',
          'typing-extensions>=3.7.2'
      ],
      extras_require=extras_require,
      test_suite='tests',
      zip_safe=False)
