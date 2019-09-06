from setuptools import setup, find_packages
setup(
      name="crnn",
      version="0.10",
      description="CRNN",
      author="piginzoo",
      url="http://www.piginzoo.com",
      license="LGPL",
      packages= find_packages(),
      scripts=["scripts/test.py"],
      )