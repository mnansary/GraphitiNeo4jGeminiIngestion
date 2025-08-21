# setup.py

from setuptools import setup, find_packages

# --- Read the contents of your README file ---
# This will be used as the long description for your package
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# --- Read the contents of your requirements file ---
# This will be used to automatically define the package's dependencies
with open("requirements.txt", "r") as f:
    install_requires = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]


setup(
    # --- Core Metadata ---
    name='graphiti-ingestion-service',
    version='0.1.0',

    # --- Author and Project Links ---
    author='mnansary',
    author_email='nazmuddoha.ansary.28@gmail.com',
    description='An asynchronous service to ingest data into a Neo4j knowledge graph using Graphiti, vLLM, and Triton.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mnansary/KnowledgeGraphNeo4j',  # URL to your project's repository

    # --- Package Discovery ---
    # find_packages() automatically finds all packages (directories with an __init__.py)
    # in your project. We can specify where to look.
    packages=find_packages(where=".", include=["graphiti_ingestion*"]),

    # --- Dependencies ---
    # This list is now dynamically read from your requirements.txt file.
    install_requires=install_requires,

    # --- Python Version Requirement ---
    # Specify the minimum version of Python required to run your project.
    python_requires='>=3.9',

    # --- Classifiers ---
    # These are standard markers that help tools like pip and PyPI categorize your project.
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
        "Framework :: FastAPI",
    ],
)