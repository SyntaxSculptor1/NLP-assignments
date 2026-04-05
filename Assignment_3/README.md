# Assignment 3 - Transformer Fine-tuning + Robustness + Limitations

<div align="center">

![Static Badge](https://img.shields.io/badge/github-repo-blue?logo=github)
![Static Badge](https://img.shields.io/badge/version-1.0.0-green)


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-212121?style=for-the-badge&logo=huggingface&logoColor=FFD21E)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
</div>


## Assignment Overview
Fine-tune a pretrained Transformer for AG News classification, compare it against your best model from Assignment 2, and perform robustness/slice evaluation. Conclude with a limitations section grounded in evidence.

Learning outcomes:
- Fine-tune a pretrained Transformer encoder for classification.
- Evaluate beyond a single score using slice/robustness analyses.
- Write scientifically grounded limitations and failure-mode reporting.

Requirements:
- Fine-tune one pretrained model (e.g., DistilBERT / BERT / RoBERTa) for 4-way classification.
- Compare against your best Assignment 2 model (same splits, same metrics).
- Report Accuracy + Macro-F1 + confusion matrix for both.
- Perform at least two robustness/slice evaluations (see options below).
- Include a short Limitations section (½–1 page) based on results and examples.

Robustness / slice evaluation (choose at least TWO):
- Length buckets: evaluate performance by input length (define bins and report per-bin metrics).
- Input field stress test: headline-only vs headline+description (if your AG News version supports both fields).
- Keyword masking probe: mask a small list of class-indicative keywords and measure performance drop.
- Label-noise sensitivity (simple): train with reduced training size (e.g., 25%, 50%, 100%) and compare trends.

Instructions:
- Reuse the same dataset split from Assignments 1–2.
- Fine-tune the Transformer (document tokenizer, max length, LR, batch size, epochs, early stopping).
- Evaluate your Transformer and your best neural model on the same test set.
- Run two slice evaluations and report results clearly (tables/plots encouraged).
- Write limitations grounded in observed failures + slice results.

Deliverables:
Final report (5–6 pages, PDF) including:
- Best baseline summary (A1/A2) + Transformer setup
- Comparison table: baseline vs neural vs Transformer (dev + test)
- Robustness/slice evaluation results + interpretation
- Error analysis (≥10 examples) emphasizing harmful/important failures
- Limitations + suggested future improvements
- Code repository with:
- Training + evaluation scripts
- Slice evaluation script/notebook
- Clear reproduction instructions
One-page Model Card (appendix or separate PDF) including intended use, metrics, known failures.

## How to run
Run the following commands in your terminal. We assume installation of the [uv package manager](https://docs.astral.sh/uv/) and that you are working on Linux/WSL/MacOS environment.

``` bash
git clone https://github.com/SyntaxSculptor1/NLP-assignments.git
cd "Assignment_3"
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
- --truncation - Number of tokens removed
- --epochs - Number of epochs to run
- --batch_size - Batch size for training CNN and LSTM
- --learningrate - How fast the gradient is adjusted
- --weight_decay - Is learning rate is scaled by root mean square

## Code reproducibility
In our report, we use the default properties which are as follows:
- Dataset - "sh0416/ag_news"
- Split - 0.1
- Seed - 67
- Verbose - True
- Padding - max_length
- Max Tokens - 10000
- Epochs - 15
- Batch size - 32
- truncation - True
- learning_rate - 2e-5
- weight_decay - 0.01

## Models created
We fine-tune 6 models with these, 3 of them have both the headlines and the description being utilized, the other 3 only have headlines. Then in each one of those 3 groups we train one with 25% of the data, another with 75% of the data and finally another with 100% of the data as to fufill the Robustness evaluations set by the assignment.

## Credits
Credits go to: [@ChrChirag](https://github.com/ChrChirag)  [@Cistaroth](https://github.com/Cistaroth) [@SyntaxSculptor1](https://github.com/SyntaxSculptor1)
