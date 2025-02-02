from setuptools import find_packages, setup
from typing import List

HYPER_E_DOT='-e.'
def get_requirements(file_path:str)->List[str]: # type: ignore
  '''
  this function will return the list of requirements
  '''

  requirements=[]
  with open(file_path) as file_obj:
    requirements=file_obj.readlines()
    requirements=[req.replace("\n","") for req in requirements]

    if HYPER_E_DOT in requirements:
      requirements.remove(HYPER_E_DOT)

  return requirements

setup(
  name='Sign-to-text',
  version='0.0.1',
  author='Aishwarya',
  author_email='bsaishwarya43@gmail.com',
  packages=find_packages(),
  install_requires=get_requirements('requirements.txt')
)
