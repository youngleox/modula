import setuptools

setuptools.setup(
    name="modula",
    packages=setuptools.find_packages(),
    version="0.0.1",
    author="anon",
    author_email="anon@anon.com",
    description="modula pytorch",
    url="git@github.com:anon/modula.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
            "torch>=2.0.0",
    ],
    python_requires='>=3.9',
)