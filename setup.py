from setuptools import find_packages, setup


def readme():
    with open("README.md", encoding="utf-8") as f:
        return f.read()


# read version file
exec(open('alibi/version.py').read())

extras_require = {
    'ray': ['ray>=0.8.7, <3.0.0'],
    # shap is separated due to build issues, see https://github.com/slundberg/shap/pull/1802
    'shap': [
        'shap>=0.40.0, <0.42.0',  # versioning: https://github.com/SeldonIO/alibi/issues/333
        'numba>=0.50.0, !=0.54.0, <0.57.0',  # Avoid 0.54 due to: https://github.com/SeldonIO/alibi/issues/466
    ],
    'tensorflow': ['tensorflow>=2.0.0, !=2.6.0, !=2.6.1, <2.11.0'],
    'torch': ['torch>=1.9.0, <2.0.0'],
    'all': [
        'ray>=0.8.7, <3.0.0',
        'shap>=0.40.0, <0.42.0',
        'numba>=0.50.0, !=0.54.0, <0.57.0',
        'tensorflow>=2.0.0, !=2.6.0, !=2.6.1, <2.11.0',
        'torch>=1.9.0, <2.0.0'
    ]
}

if __name__ == '__main__':
    setup(name='alibi',
          author='Seldon Technologies Ltd.',
          author_email='hello@seldon.io',
          version=__version__,  # type: ignore # noqa F821
          description='Algorithms for monitoring and explaining machine learning models',
          long_description=readme(),
          long_description_content_type='text/markdown',
          url='https://github.com/SeldonIO/alibi',
          license="Apache 2.0",
          packages=find_packages(),
          include_package_data=True,
          python_requires='>=3.7',
          # lower bounds based on Debian Stable versions where available
          install_requires=[
              'numpy>=1.16.2, <2.0.0',
              'pandas>=0.23.3, <2.0.0',
              'scikit-learn>=1.0.0, <2.0.0',
              'spacy[lookups]>=2.0.0, <4.0.0',
              'blis<0.8.0',  # Windows memory issues https://github.com/explosion/thinc/issues/771
              'scikit-image>=0.17.2, <0.20',  # introduced `start_label` argument for `slic`
              'requests>=2.21.0, <3.0.0',
              'Pillow>=5.4.1, <10.0',
              'attrs>=19.2.0, <23.0.0',
              'scipy>=1.1.0, <2.0.0',
              'matplotlib>=3.0.0, <4.0.0',
              'typing-extensions>=3.7.4.3',
              'dill>=0.3.0, <0.4.0',
              'transformers>=4.7.0, <5.0.0',
              'tqdm>=4.28.1, <5.0.0',
          ],
          extras_require=extras_require,
          test_suite='tests',
          zip_safe=False,
          classifiers=[
              "Intended Audience :: Science/Research",
              "Operating System :: OS Independent",
              "Programming Language :: Python :: 3",
              "Programming Language :: Python :: 3.7",
              "Programming Language :: Python :: 3.8",
              "Programming Language :: Python :: 3.9",
              "Programming Language :: Python :: 3.10",
              "License :: OSI Approved :: Apache Software License",
              "Topic :: Scientific/Engineering",
          ])
