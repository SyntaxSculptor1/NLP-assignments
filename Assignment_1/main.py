from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse

from load_data import load_nltk_models, load_data
from preprocess_data import merge_title_description, text_cleaning, tfidf_transform
from logistic_regression import (
    perform_logistic_regression,
    evaluate_logistic_regression,
)
from svm import perform_support_vector_machine, evaluate_support_vector_machine


def argument_parsing() -> argparse.Namespace:
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
    args = argument_parsing()

    DATASET = args.dataset
    VERBOSE = args.verbose
    SPLIT = args.split
    SEED = args.seed

    load_nltk_models()

    train_data, validation_data, test_data = load_data(
        dataset=DATASET, split=SPLIT, seed=SEED
    )

    train_data, validation_data, test_data = merge_title_description(
        datasets=[train_data, validation_data, test_data],
        title_column="title",
        description_column="description",
        new_column="text",
    )

    lemmatizer = WordNetLemmatizer()
    train_data, validation_data, test_data = text_cleaning(
        datasets=[train_data, validation_data, test_data],
        column="text",
        lemmatizer=lemmatizer,
    )

    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")

    x_train, x_val, x_test = tfidf_transform(
        train_texts=train_data["text"],
        validation_texts=validation_data["text"],
        test_texts=test_data["text"],
        vectorizer=vectorizer,
    )

    y_train, y_val, y_test = (
        train_data["label"],
        validation_data["label"],
        test_data["label"],
    )

    logreg_param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
    logreg_grid_search = perform_logistic_regression(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        param_grid=logreg_param_grid,
        scoring="accuracy",
        max_iter=5000,
        verbose=int(VERBOSE),
        seed=SEED,
    )

    evaluate_logistic_regression(
        model=logreg_grid_search.best_estimator_, x_test=x_test, y_test=y_test
    )

    svm_param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}

    print("PRE SVM ")
    svm_grid_search = perform_support_vector_machine(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        param_grid=svm_param_grid,
        scoring="accuracy",
        verbose=int(VERBOSE),
    )

    print("POST SVM ")
    evaluate_support_vector_machine(
        model=svm_grid_search.best_estimator_, x_test=x_test, y_test=y_test
    )

    print("POST EVAL")
if __name__ == "__main__":
    main()
