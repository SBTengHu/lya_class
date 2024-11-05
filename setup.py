
#
# Standard imports
#
import sys
import glob, os
from setuptools import setup, find_packages
setup_keywords = dict()
setup_keywords['name'] = 'lya_class'
setup_keywords['description'] = 'structure function codes'
setup_keywords['author'] = 'T. Hu'
#setup_keywords['author_email'] = 'teng.hu@lam.fr'
setup_keywords['license'] = 'BSD'
setup_keywords['url'] = 'https://github.com/SBTengHu/lya_class'
setup_keywords['version'] = '0.0.dev0'

setup_keywords['long_description'] = ''
if os.path.exists('README.md'):
    with open('README.md') as readme:
        setup_keywords['long_description'] = readme.read()


if os.path.isdir('bin'):
    setup_keywords['scripts'] = [fname for fname in glob.glob(os.path.join('bin', '*'))
        if not os.path.basename(fname).endswith('.rst')]
setup_keywords['provides'] = [setup_keywords['name']]
setup_keywords['requires'] = ['Python (>3.0.0)']
setup_keywords['zip_safe'] = False
setup_keywords['use_2to3'] = False
setup_keywords['packages'] = find_packages()
#setup_keywords['package_dir'] = {'':''}
#setup_keywords['cmdclass'] = {'version': DesiVersion, 'test': DesiTest, 'sdist': DistutilsSdist}
#setup_keywords['test_suite']='{name}.tests.{name}_test_suite.{name}_test_suite'.format(**setup_keywords)
setup_keywords['setup_requires']=['pytest-runner']
setup_keywords['tests_require']=['pytest']
#setup_keywords['cmdclass']={'build_ext': build_ext}


