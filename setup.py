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
