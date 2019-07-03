import logging

from sklearn.metrics import make_scorer, fbeta_score

from classes.Dataset import Dataset
from classes.EnsembleVotingPredictor import EnsembleVotingPredictor
from classes.Example import Example
from classes.Initializer import Initializer
from classes.Preprocessor import Preprocessor
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
    one_hot_encoded = ['sec_struct', 'amino_type']
    not_to_be_slided = ['chain_len', 'struct_-', 'struct_B',
                        'struct_S', 'struct_T', 'struct_E',
                        'struct_H', 'struct_I', 'struct_G']

    dataset_preprocessor = Preprocessor(dataset.get_features(), dataset.get_residues_info())
    encoders = dataset_preprocessor.apply_one_hot_encoding(one_hot_encoded)
    dataset_preprocessor.fill_missing_values(exclude=one_hot_encoded)
    dataset_preprocessor.apply_sliding_window(to_be_excluded=not_to_be_slided,
                                              k=data_processing_params["sw_width"])

    # Update the dataset features post preprocessing
    dataset.update_features(dataset_preprocessor.get_features())

    # Shuffle
    dataset.shuffle_dataset()

    # Balance
    # dataset.balance(dataset_params["balance_ratio"])

    # Get the final features and labels to perform the training of the models
    features = list(dataset.get_features().columns)

    x = dataset.get_features(features)
    y = dataset.get_labels()

    # ---- PREDICTOR INITIALIZATION ----

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
                      delete_previous_example=True)

    # ---- EXAMPLE PREPROCESSING ----

    # Perform basic preprocessing on the example
    example_preprocessor = Preprocessor(example.get_features(), example.get_residues_info())

    example_preprocessor.apply_one_hot_encoding(to_be_encoded=one_hot_encoded, encoders=encoders)
    example_preprocessor.fill_missing_values()
    example_preprocessor.apply_sliding_window(to_be_excluded=not_to_be_slided,
                                              k=data_processing_params["sw_width"])
    example_preprocessor.fill_missing_values()
    # Update the dataset features post preprocessing
    example.update_features(example_preprocessor.get_features())

    # ---- PREDICTION ----

    # Predict results using trained model
    example_processed = example.get_features(features)
    results = clf.predict_proba(example_processed)

    export_results(results, example.get_structure(), initializer.get_out_dir())

    logging.info("Results successfully generated in {}/out_{}.txt"
                 .format(initializer.get_out_dir(), example.get_structure().id))


if __name__ == '__main__':
    main()
