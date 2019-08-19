import os,re
import setuptools

PACKAGE_NAME    = "gecosistema_gdal"
AUTHOR          = "Valerio Luzzi"
EMAIL           = "valluzzi@gmail.com"
GITHUB          = "https://github.com/valluzzi/%s.git"%(PACKAGE_NAME)
DESCRIPTION     = "A core functions package"

def get_version():
    VERSIONFILE = os.path.join(PACKAGE_NAME, '__init__.py')
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]\s*"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version string in %s.' % (VERSIONFILE,))

setuptools.setup(
    name=PACKAGE_NAME,
    version=get_version(),
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    url=GITHUB,
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    install_requires=[]
)
