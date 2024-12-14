from setuptools import find_packages, setup

setup(
    name='chewbite_fusion',
    packages=['chewbite_fusion'],
    package_dir={'': 'src'},
    version='0.1.0',
    description='Audio and movement events detection.',
    author='sinc(i)',
    license='MIT',
    install_requires=[
        'pandas==1.5.3',
        'plotly==5.14.0',
        'scipy==1.10.1',
        'notebook==6.5.3',
        'scikit-learn==1.2.2',
        'librosa==0.10.0',
        'more_itertools==9.1.0',
        'yaer @ git+https://github.com/arielrossanigo/yaer.git#egg=yaer',
        'tensorflow==2.12.0',
        'sed_eval==0.2.1',
        'xgboost==1.7.5',
        'seaborn==0.12.2'
    ]
)
