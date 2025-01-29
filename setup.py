from setuptools import setup, find_packages

setup(
    name="kokoro-onnx-flask",
    version="0.1",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,  # Include non-code files specified in MANIFEST.in
    package_data={
        "kokoro_onnx_flask": ["static/*", "templates/*"],  # Specify file patterns
    },
    install_requires=[
        "flask>=3.1.0",
        "kokoro-onnx>=0.3.6",
        "numpy>=2.0.2",
        "scipy>=1.15.1",
        "spacy>=3.8.4",
    ],
    # Optional metadata
    author="Dave1475",
    description="Flask server wrapper for kokoro-onnx",
    url="https://github.com/Dave1475/kokoro-onnx-flask",
)
