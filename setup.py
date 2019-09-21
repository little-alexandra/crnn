from setuptools import setup, find_packages
setup(
	name="crnn",
	version="1.0",
	description="ocr crnn module",
	author="piginzoo",
	url="http://www.piginzoo.com",
	license="LGPL",
	packages=find_packages(where='.', exclude=(), include=('*',)),
	package_dir={'crnn.config': 'crnn/config'},
	package_data={'crnn.config':['charset.3770.txt','charset.5987.txt','charset.6883.txt']}
)