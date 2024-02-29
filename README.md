# Demo Repository for Bias Detection in Embeddings

## Overview

This repository is dedicated to the exploration and detection of biases within various embedding models. Our goal is to assess and mitigate potential biases in machine learning embeddings, ensuring fairer and more equitable outcomes in AI applications. This is a test repository where we experiment with different methodologies and techniques to identify and address bias within data representations.

## Objectives

- **Identify Bias**: Utilize analytical tools and techniques to uncover inherent biases within embeddings.
- **Assessment**: Evaluate the impact and extent of detected biases on model performance and decision-making.
- **Mitigation Strategies**: Develop and test strategies to mitigate identified biases, aiming to improve fairness and accuracy.
- **Community Engagement**: Foster a community of researchers and practitioners focused on bias detection and mitigation in AI.

## Installation

Instructions on how to set up the project environment:

## Installation

Follow these instructions to set up the project environment:

### Prerequisites:
Ensure you have **Python 3.11** and **Poetry** installed on your system.

#### Python Installation:
- **Windows**: Visit [Python Releases for Windows](https://www.python.org/downloads/windows/) and download the latest installer for Python 3.11.
- **macOS**: Visit [Python Releases for Mac OS X](https://www.python.org/downloads/mac-osx/) and download the latest installer for Python 3.11.

#### Poetry Installation:
Poetry is a tool for dependency management and packaging in Python. To install Poetry, follow the instructions on the [official Poetry website](https://python-poetry.org/docs/#installing-with-the-official-installer).

### Setting up the Project Environment:
After installing the prerequisites, execute the following commands in your terminal:

```bash
poetry init
poetry shell
poetry install
streamlit run src/digdir/vectorEmbedings/visualize_embeddings.py 
```

These commands will:

1. Initialize a new poetry environment for your project.
2. Activate the poetry shell, creating an isolated environment for your dependencies.
3. Install the necessary dependencies for the project specified in the `pyproject.toml` file.
4. Run the Streamlit app located at `src/digdir/vectorEmbedings/visualize_embeddings.py`.

Remember to navigate to your project directory before executing these commands.

## Contributing

We welcome contributions from the community. Please read our CONTRIBUTING.md guide for details on how to submit changes and suggestions.

## License

Available to the public.
