from setuptools import find_packages, setup


def readme():
    with open('README.md') as f:
        return f.read()


# read version file
exec(open('alibi/version.py').read())


def get_extra_requires(path, add_all=True):
    """
    Reads and inverts the requirements in requirements/extra.txt
    """
    import re
    from collections import defaultdict

    with open(path) as fp:
        extra_deps = defaultdict(set)
        for k in fp:
            if k.strip() and not k.startswith('#'):
                tags = set()
                if ':' in k:
                    k, v = k.split(':')
                    tags.update(vv.strip() for vv in v.split(','))
                tags.add(re.split('[<=>]', k)[0])
                for t in tags:
                    extra_deps[t].add(k)

        # add tag `all` at the end
        if add_all:
            extra_deps['all'] = set(vv for v in extra_deps.values() for vv in v)

    return extra_deps


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
              'scikit-learn>=0.20.2, <1.1.0',
              'spacy[lookups]>=2.0.0, <4.0.0',
              'scikit-image>=0.14.2, !=0.17.1, <0.20',  # https://github.com/SeldonIO/alibi/issues/215
              'requests>=2.21.0, <3.0.0',
              'Pillow>=5.4.1, <10.0',
              'attrs>=19.2.0, <22.0.0',
              'scipy>=1.1.0, <2.0.0',
              'matplotlib>=3.0.0, <4.0.0',
              'typing-extensions>=3.7.4.3',
              'dill>=0.3.0, <0.4.0',
              'transformers>=4.7.0, <5.0.0',
              'tqdm>=4.28.1, <5.0.0'
          ],
          classifiers=[
              "Intended Audience :: Science/Research",
              "Operating System :: OS Independent",
              "Programming Language :: Python :: 3",
              "Programming Language :: Python :: 3.7",
              "Programming Language :: Python :: 3.8",
              "Programming Language :: Python :: 3.9",
              "License :: OSI Approved :: Apache Software License",
              "Topic :: Scientific/Engineering",
          ],
          extras_require = get_extra_requires('requirements/extra.txt'),
          test_suite = 'tests',
          zip_safe = False)