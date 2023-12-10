import random

from src.recommender_core.data_handling.evaluation.evaluation_results import get_eval_results_header


def random_hyperparameter_choice(model_variants, vector_size_range, window_range, min_count_range,
                                 epochs_range, sample_range, negative_sampling_variants):
    model_variant = random.choice(model_variants)
    vector_size = random.choice(vector_size_range)
    window = random.choice(window_range)
    min_count = random.choice(min_count_range)
    epochs = random.choice(epochs_range)
    sample = random.choice(sample_range)
    negative_sampling_variant = random.choice(negative_sampling_variants)
    return model_variant, vector_size, window, min_count, epochs, sample, negative_sampling_variant


def prepare_hyperparameters_grid():
    # example hyparparams
    negative_sampling_variants = range(5, 20, 5)  # 0 = no negative sampling
    no_negative_sampling = 0  # use with hs_soft_max
    vector_size_range = range(50, 450, 50)
    window_range = [1, 2, 4, 5, 8, 12, 16, 20]
    min_count_range = [0, 1, 2, 3, 5, 8, 12]
    epochs_range = [20, 25, 30]
    sample_range = [0.0, 1.0 * (10.0 ** -1.0), 1.0 * (10.0 ** -2.0), 1.0 * (10.0 ** -3.0), 1.0 * (10.0 ** -4.0),
                    1.0 * (10.0 ** -5.0)]

    corpus_title, model_results = get_eval_results_header()
    # noinspection PyPep8
    return negative_sampling_variants, no_negative_sampling, vector_size_range, window_range, \
        min_count_range, epochs_range, sample_range, corpus_title, model_results
