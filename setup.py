from setuptools import setup, find_packages

setup(
    name="kokoro-onnx-flask",  # Project name
    version="0.1",             # Version of the project
    packages=find_packages(),  # Automatically find packages
    install_requires=[         # List of dependencies
        "flask",
        "kokoro-onnx",  # Assuming kokoro-onnx is a separate installable package
        "numpy",
        "scipy",
        "spacy",
    ],
    # Optional metadata
    author="Dave1475",
    description="Flask server wrapper for kokoro-onnx",
    url="https://github.com/Dave1475/kokoro-onnx-flask",
)
