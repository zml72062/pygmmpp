from distutils.core import setup

setup(
    name='pygmmpp',
    description='PyG Minus Minus Plus Plus',
    author='zml72062',
    packages=['pygmmpp',
              'pygmmpp.data',
              'pygmmpp.data.sys',
              'pygmmpp.datasets',
              'pygmmpp.datasets.io',
              'pygmmpp.nn',
              'pygmmpp.utils'],
    package_data={'pygmmpp.datasets': ['master.csv']}
)
