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
        "flask>=3.1.0,<4.0",
        "kokoro-onnx>=1.0.0,<2.0",
        "numpy>=2.0.2,<3.0",
        "scipy>=1.15.1,<2.0",
        "spacy>=3.8.4,<4.0",
    ],
    # Optional metadata
    author="Dave1475",
    description="Flask server wrapper for kokoro-onnx",
    url="https://github.com/Dave1475/kokoro-onnx-flask",
)
