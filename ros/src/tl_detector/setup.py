from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
	packages=['core', 'protos', 'utils'],
	package_dir={'': 'src'})

setup(**setup_args)

setup_args = generate_distutils_setup(
	packages=['light_classification'])

setup(**setup_args)