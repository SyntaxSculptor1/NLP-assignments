from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse

from load_data import load_nltk_models, load_data
from preprocess_data import merge_title_description, text_cleaning, tfidf_transform
from models import scale_data_and_define_split, perform_logistic_regression, perform_support_vector_machine
from evaluation import evaluate_model, find_misclassified
from rich.console import Console
from rich.panel import Panel

console = Console()

def argument_parsing() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="sh0416/ag_news",
        type=str,
        help="Dataset to use. Default: sh0416/ag_news",
    )

    parser.add_argument(
        "--verbose", default=True, type=bool, help="Verbose mode. Default: True"
    )

    parser.add_argument(
        "--split", default=0.1, type=float, help="Split ratio. Default: 0.1"
    )

    parser.add_argument("--seed", default=67, type=int, help="Random seed. Default: 67")

    args = parser.parse_args()

    return args


def main() -> None:
    """
    Main function
    
    Returns:
        None
    """

    program_title = Panel(
        "[bold white] Assignment 1 - News Classification using TF-IDF features [/bold white]"
    )
    console.print(program_title)

    args = argument_parsing()

    DATASET = args.dataset
    VERBOSE = args.verbose
    SPLIT = args.split
    SEED = args.seed

    load_nltk_models(verbose=VERBOSE)

    train_data, validation_data, test_data_raw = load_data(
        dataset=DATASET, split=SPLIT, seed=SEED, verbose=VERBOSE
    )

    train_data, validation_data, test_data_raw = merge_title_description(
        datasets=[train_data, validation_data, test_data_raw],
        title_column="title",
        description_column="description",
        new_column="text",
        verbose=VERBOSE
    )

    lemmatizer = WordNetLemmatizer()
    train_data, validation_data, test_data = text_cleaning(
        datasets=[train_data, validation_data, test_data_raw],
        column="text",
        lemmatizer=lemmatizer,
        verbose=VERBOSE
    )

    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")

    x_train, x_val, x_test = tfidf_transform(
        train_texts=train_data["text"],
        validation_texts=validation_data["text"],
        test_texts=test_data["text"],
        vectorizer=vectorizer,
        verbose=VERBOSE
    )

    y_train, y_val, y_test = (
        train_data["label"],
        validation_data["label"],
        test_data["label"],
    )

    x_full, y_full, predefined_split = scale_data_and_define_split(
        x_train= x_train,
        x_val= x_val,
        y_train= y_train,
        y_val= y_val,
        verbose=VERBOSE
    )

    logreg_param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
    logreg_model = perform_logistic_regression(
        x_full=x_full,
        y_full=y_full,
        predefined_split=predefined_split,
        param_grid=logreg_param_grid,
        scoring="accuracy",
        max_iter=5000,
        verbose=VERBOSE,
        seed=SEED,
    )

    evaluate_model(
        model=logreg_model,
        model_name="Logistic Regression",
        x_test=x_test,
        y_test=y_test,
        verbose=VERBOSE,
        plot=True
    )

    find_misclassified(
        model=logreg_model,
        model_name="Logistic Regression",
        x_test_raw=test_data_raw,
        x_test_cleaned=test_data,
        x_test=x_test,
        y_test=y_test,
        verbose=VERBOSE,
        save=True
    )

    svm_param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}

    svm_model = perform_support_vector_machine(
        x_full=x_full,
        y_full=y_full,
        predefined_split=predefined_split,
        param_grid=svm_param_grid,
        scoring="accuracy",
        verbose=VERBOSE,
    )

    evaluate_model(
        model=svm_model,
        model_name="Support Vector Machine",
        x_test=x_test,
        y_test=y_test,
        verbose=VERBOSE,
        plot=True
    )

    find_misclassified(
        model=svm_model,
        model_name="Support Vector Machine",
        x_test_raw=test_data_raw,
        x_test_cleaned=test_data,
        x_test=x_test,
        y_test=y_test,
        verbose=VERBOSE,
        save=True
    )

if __name__ == "__main__":
    main()
