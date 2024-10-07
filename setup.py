from setuptools import setup, find_packages

print("Start Mathematical-Optimization setuptools...")
setup(
    name='Mathematical-Optimization',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy', 'cvxopt'
    ],
    test_suite='tests',
)

# Installation des Pakets:
# python setup.py install

# Erstellen eines Quellverteilungspakets:
# python setup.py sdist

# Erstellen eines Binärverteilungspakets:
# python setup.py bdist

# Installation des Pakets im Entwicklungsmodus:
# python setup.py develop