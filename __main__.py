import logging

from Bio.PDB import is_aa
from pandas import DataFrame
from sklearn.utils import shuffle

from classes.Dataset import Dataset
from classes.EnsembleVotingPredictor import EnsembleVotingPredictor
from classes.Example import Example
from classes.FeaturesSelector import FeaturesSelector
from classes.Initializer import Initializer
from classes.Preprocessor import Preprocessor
from sklearn.metrics import fbeta_score, make_scorer

from functions.export_results import export_results


def main():
    # ---- INITIALIZATION ----

    # Take arguments from input and initialize logger
    initializer = Initializer()

    # Initialize the input pdb id
    pdb_id = initializer.get_pdb_id()

    # Initialize the dssp path
    dssp_path = initializer.get_dssp_path()

    # Initialize the parameters from params.json
    params = initializer.get_params()
    dataset_params = params.dataset
    session_params = params.session
    data_processing_params = params.data_processing
    svm_params = params.svm
    forest_params = params.forest
    logistic_params = params.logistic

    # Initialize the dataset
    dataset = Dataset(dssp_path,
                      rewrite=dataset_params["rewrite"],
                      use_ring_api=dataset_params["use_ring_api"],
                      delete_edges=dataset_params["delete_edges"],
                      delete_entities=dataset_params["delete_entities"])

    # ---- DATASET PREPROCESSING ----

    # Perform basic preprocessing on features (fill missing and scale values, averaging sliding window)
    dataset_preprocessor = Preprocessor(dataset.get_features())
    dataset_preprocessor.apply_features_scaling(dataset.get_features_names())
    to_be_preprocessed = dataset.get_features_names()
    to_be_preprocessed.remove("structural_linearity")
    dataset_preprocessor.apply_sliding_window(to_be_preprocessed,
                                              data_processing_params["sw_width"])

    # Update the dataset features post preprocessing
    dataset.update_features(dataset_preprocessor.get_features())

    # Balance the dataset positive and negative examples ratio
    dataset.balance(dataset_params["balance_ratio"])

    # ---- DATASET FEATURES SELECTION ----

    # Perform feature analysis and selection
    feature_selector = FeaturesSelector(dataset)
    feature_selector.univariate_selection(num_features=data_processing_params["num_features"])
    feature_selector.features_importance(num_features=data_processing_params["num_features"],
                                         show=data_processing_params["show_best_features"])

    # Update the dataset features post feature selection
    dataset.update_features(feature_selector.get_features())

    # ---- PREDICTOR INITIALIZATION ----

    # Get the final features and labels to perform the training of the models
    df = dataset.get_dataset()

    # Shuffle the dataset
    x: DataFrame = shuffle(df)
    y = x.pop('is_lip')

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    scorer = make_scorer(fbeta_score, beta=0.7, pos_label=1)

    # Initialize a predictor
    clf = EnsembleVotingPredictor(x, y, scoring_f=scorer)

    if session_params["load_clf"] is True:
        # Load pre-trained model from file (for speed)
        clf.load()
    else:
        # Select the appropriate models based on the session parameters
        if session_params["forest"] is True:
            clf.add_random_forest(forest_params)
        if session_params["svm"] is True:
            clf.add_svm(svm_params)
        if session_params["logistic"] is True:
            clf.add_logistic_classifier(logistic_params)

        # Train model and save it to file (overwrite if it already exists)
        clf.fit()
        if session_params["dump_clf"] is True:
            # Save to file if required
            clf.dump()

    clf.evaluate(k_fold=8)

    # ---- EXAMPLE INITIALIZATION ----

    # Create an example to be predicted from the input parameters
    example = Example(pdb_id,
                      dssp_path,
                      use_ring_api=dataset_params["use_ring_api"],
                      delete_previous_example=session_params["delete_previous_input"])

    # ---- EXAMPLE PREPROCESSING ----

    # Perform basic preprocessing on the example (fill missing and scale values, averaging sliding
    # window)
    example_preprocessor = Preprocessor(example.get_features())
    example_preprocessor.apply_features_scaling(example.get_features_names())
    example_preprocessor.apply_sliding_window(to_be_preprocessed,
                                              data_processing_params["sw_width"])

    # Update the dataset features post preprocessing
    example.update_features(example_preprocessor.get_features())

    # ---- EXAMPLE FEATURES SELECTION ----

    # Remove all non-best features from the example
    best_features_labels = feature_selector.get_best_features_ids()

    # ---- PREDICTION ----

    # Predict results using trained model
    results = clf.predict_proba(example.get_features(best_features_labels))

    export_results(results, example.get_structure(), initializer.get_out_dir())

    logging.info("Results successfully generated in {}/out_{}.txt"
                 .format(initializer.get_out_dir(), example.get_structure().id))


if __name__ == '__main__':
    main()
