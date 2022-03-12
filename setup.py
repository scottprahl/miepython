import re
import os.path
from setuptools import setup

project = 'miepython'


def get_init_property(prop):
    """Return property from __init__.py."""
    here = os.path.abspath(os.path.dirname(__file__))
    file_name = os.path.join(here, project, '__init__.py')
    regex = r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop)
    with open(file_name, 'r', encoding='utf-8') as file:
        result = re.search(regex, file.read())
    return result.group(1)


def get_contents(filename):
    """Return contents of filename relative to the location of this file."""
    here = os.path.abspath(os.path.dirname(__file__))
    fn = os.path.join(here, filename)
    with open(fn, 'r', encoding='utf-8') as f:
        contents = f.read()
    return contents


setup(
    name=project,
    long_description=get_contents('README.rst'),
    long_description_content_type='text/x-rst',
    version=get_init_property('__version__'),
    author=get_init_property('__author__'),
    author_email=get_init_property('__email__'),
    license=get_init_property('__license__'),
    url=get_init_property('__url__')
)
