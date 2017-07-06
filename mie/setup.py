from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='mie',
      version='0.1',
      description='Mie scattering from spheres',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Topic :: Text Processing :: Linguistic',
      ],
      keywords='Mie scattering water droplets',
      url='http://github.com/storborg/funniest',
      author='Scott Prahl',
      author_email='scott.prahl@oit.edu',
      license='MIT',
      packages=['mie'],
      install_requires=[
          'markdown',
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      entry_points={
          'console_scripts': ['funniest-joke=funniest.command_line:main'],
      },
      include_package_data=True,
      zip_safe=False)