from setuptools import setup, find_packages

setup(
    name="kokoro-onnx-flask",
    version="0.1",
    packages=find_packages(where='src'),  # Look for packages inside 'src'
    package_dir={'': 'src'},  # Tell setuptools where to find your packages
    install_requires=[
        "flask",
        "kokoro-onnx",
        "numpy",
        "scipy",
        "spacy",
    ],
    # Optional metadata
    author="Dave1475",
    description="Flask server wrapper for kokoro-onnx",
    url="https://github.com/Dave1475/kokoro-onnx-flask",
)
