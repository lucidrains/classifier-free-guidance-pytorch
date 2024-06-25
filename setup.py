from setuptools import setup, find_packages

setup(
  name = 'classifier-free-guidance-pytorch',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  version = '0.6.8',
  license='MIT',
  description = 'Classifier Free Guidance - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/classifier-free-guidance-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'classifier free guidance',
    'text conditioning and guidance'
  ],
  install_requires=[
    'beartype',
    'einops>=0.8',
    'environs',
    'ftfy',
    'open-clip-torch>=2.8.0',
    'torch>=2.0',
    'transformers[torch]'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
