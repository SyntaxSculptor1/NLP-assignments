# Assignment 1 - Classical Baselines + Evaluation Pack

<div align="center">

![Static Badge](https://img.shields.io/badge/github-repo-blue?logo=github)
![Static Badge](https://img.shields.io/badge/version-1.0.0-green)


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NLTK](https://img.shields.io/badge/nltk-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-212121?style=for-the-badge&logo=huggingface&logoColor=FFD21E)
</div>

## Assignment Overview
Build strong classical baselines for AG News, evaluate them rigorously, and document findings with an error analysis. This assignment establishes the experimental workflow used in later assignments.

Learning outcomes:
- Implement a reproducible text classification pipeline (data → features → model → evaluation).
- Compare baseline models fairly with appropriate metrics.
- Perform structured error analysis and interpret confusion patterns.

Requirements:
- Use word-level TF-IDF features (document the preprocessing choices).
- Train two classical models (required):
    - TF-IDF + Logistic Regression
    - TF-IDF + Linear SVM
- Report Accuracy + Macro-F1 + confusion matrix.

Instructions:

1. Load AG News and create train/dev/test splits (dev from train).
2. Implement preprocessing (tokenization, normalization) and document it.
3. Train both baseline models. Models should be trained using the description only or both title and description, but not the title alone. Keep the dev split for model selection/tuning.
4. Evaluate both models on test once for the final numbers. 
5. Present ≥20 misclassified examples from test set for each model.
6. Conduct error analysis for the best-performing model on ≥10 on misclassified examples

Deliverables:
- Report (2–3 pages, PDF) including:
    - Dataset + split details (sizes, seed)
    - Baseline models + hyperparameters
    - Results table (dev + test) + confusion matrix
    - Error analysis (≥10 examples) + insights
    - Reproducibility notes (how to run)
- Code repository with:
    - requirements.txt or environment file
    - One command to reproduce main results
    - Clear folder structure
## How to run
Run the following commands in your terminal. We assume installation of the [uv package manager](https://docs.astral.sh/uv/) and that you are working on Linux/WSL/MacOS environment.

``` bash
git clone https://github.com/SyntaxSculptor1/NLP-assignments.git
cd "Assignment_1"
uv sync
uv run main.py
```

We allow for argument parsing, these arguments include:
- --dataset - The dataset to use
- --split - The train-validation split
- --seed - The seed for code reproducibility
- --verbose - Setting verbosity

## Code reproducibility
In our report, we use the default properties which are as follows:
- Dataset - "sh0416/ag_news"
- Split - 0.1
- Seed - 67
- Verbose - True

## Credits
Credits go to: [@ChrChirag](https://github.com/ChrChirag)  [@Cistaroth](https://github.com/Cistaroth) [@SyntaxSculptor1](https://github.com/SyntaxSculptor1)