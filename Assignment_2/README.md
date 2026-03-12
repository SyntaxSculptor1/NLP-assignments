# Assignment 2 - Neural Model Comparison + Ablation

<div align="center">

![Static Badge](https://img.shields.io/badge/github-repo-blue?logo=github)
![Static Badge](https://img.shields.io/badge/version-1.0.0-green)


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-212121?style=for-the-badge&logo=huggingface&logoColor=FFD21E)
</div>

## Assignment Overview
Implement two neural text classifiers for AG News (CNN and LSTM), compare them under controlled conditions, and run a small ablation study to test a specific hypothesis about representation or training.

Learning outcomes:
- Implement neural encoders for text classification.
- Run controlled comparisons and interpret learning curves.
- Design and analyze an ablation that isolates one factor.

Requirements:
- Train two neural models (required):
    - CNN text classifier (convolution + pooling)
    - LSTM (or BiLSTM) classifier (sequence encoder + pooling)
- Controlled setup: same splits, same tokenizer, same max length, same metrics.
- Report Accuracy + Macro-F1 + confusion matrix for both models.
- Include learning curves (train loss + dev metric) for both models.

Ablation (choose ONE, required):
- **Embeddings**: pretrained vs random initialization
- **Max length**: e.g., 64 vs 128 vs 256 tokens
- **Regularization**: dropout 0 vs 0.3 (or another controlled value)
- **Capacity**: hidden size small vs medium (keep all else fixed)

Instructions:

1. Reuse the same AG News split from Assignment 1.
2. Implement CNN and LSTM models with documented hyperparameters.
3. Train using dev for early stopping / selection.
4. Run the ablation by changing only the selected factor.
5. Evaluate on test once for final numbers.
6. Update error analysis: ≥10 errors, highlight differences vs Assignment 1 baseline failures.

Deliverables:
- Report (3–4 pages, PDF) including:
    - Model descriptions (diagrams optional but encouraged)
    - Hyperparameters + training protocol (optimizer, LR, batch size, epochs, early stopping)
    - Results table + confusion matrices
    - Learning curves + interpretation (overfitting/underfitting)
    - Ablation setup + results + conclusion
    - Error analysis + comparison to Assignment 1
- Code repository with training + evaluation scripts/notebooks and reproducible runs.

## How to run
Run the following commands in your terminal. We assume installation of the [uv package manager](https://docs.astral.sh/uv/) and that you are working on Linux/WSL/MacOS environment.

``` bash
git clone https://github.com/SyntaxSculptor1/NLP-assignments.git
cd "Assignment_2"
uv sync
uv run main.py
```

We allow for argument parsing, these arguments include:
- --dataset - The dataset to use
- --split - The train-validation split
- --seed - The seed for code reproducibility
- --verbose - Setting verbosity
- --padding - The padding of text sequences
- --max_tokens - Maximum number of unique tokens allowed to pass
- --epochs - Number of epochs to run
- --batch_size - Batch size for training CNN and LSTM

## Code reproducibility
In our report, we use the default properties which are as follows:
- Dataset - "sh0416/ag_news"
- Split - 0.1
- Seed - 67
- Verbose - True
- Padding - 100
- Max Tokens - 10000
- Epochs - 100
- Batch size - 32

The model architecture of the CNN is as follows:
- Vectorizer Layer -> Embedding layer -> 1D Convolutional layer -> 1D Global Max Pooling layer -> 1 Dense Layer -> 1 Dropout layer -> 1 Classification Head

- Vectorizer layer:
    - Max Tokens - 10000
    - Padding - 100
- Embedding layer:
    - Vocab size - 10000
    - Embedding size - 100
- 1D Convolutional layer:
    - Number of filters - 128
    - Kernel Size - 5
    - Activation Function - "relu"
- 1D Global Max Pooling layer:
    - No parameters
- Dense layer:
    - Size - 64
    - Activation Function - "relu"
- Dropout layer:
    - Dropout layer was variable, either 0 or 0.3 (ablation)
- Classification head:
    - Size - 4
    - Activation function: "softmax"

The model architecture of the LSTM is as follows:
- Vectorizer Layer -> Embedding layer -> 1 Bidirectional LSTM layer -> 1 Dense layer -> 1 Dropout layer -> 1 Classification Head

- Vectorizer layer:
    - Max Tokens - 10000
    - Padding - 100
- Embedding layer:
    - Vocab size - 10000
    - Embedding size - 100
- LSTM layer:
    - Size - 64
- Dense layer:
    - Size - 64
    - Activation Function - "relu"
- Dropout layer:
    - Dropout layer was variable, either 0 or 0.3 (ablation)
- Classification head:
    - Size - 4
    - Activation function: "softmax"

## Credits
Credits go to: [@ChrChirag](https://github.com/ChrChirag)  [@Cistaroth](https://github.com/Cistaroth) [@SyntaxSculptor1](https://github.com/SyntaxSculptor1)