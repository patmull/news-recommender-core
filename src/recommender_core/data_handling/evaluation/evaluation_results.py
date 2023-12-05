from typing import Dict, List


def get_eval_results_header():
    corpus_title = ['100% Corpus']
    model_results = {'Validation_Set': [],  # type: ignore
                     'Model_Variant': [],
                     'Negative': [],
                     'Vector_size': [],
                     'Window': [],
                     'Min_count': [],
                     'Epochs': [],
                     'Sample': [],
                     'Softmax': [],
                     'Word_pairs_test_Pearson_coeff': [],
                     'Word_pairs_test_Pearson_p-val': [],
                     'Word_pairs_test_Spearman_coeff': [],
                     'Word_pairs_test_Spearman_p-val': [],
                     'Word_pairs_test_Out-of-vocab_ratio': [],
                     'Analogies_test': []
                     }  # type: Dict[str, List]
    return corpus_title, model_results


def append_training_results(source, corpus_title, model_variant, negative_sampling_variant, vector_size,
                            window,
                            min_count, epochs, sample, hs_softmax, pearson_coeff_word_pairs_eval,
                            pearson_p_val_word_pairs_eval, spearman_p_val_word_pairs_eval,
                            spearman_coeff_word_pairs_eval, out_of_vocab_ratio, analogies_eval, model_results):
    model_results['Validation_Set'].append(source + " " + corpus_title)
    model_results['Model_Variant'].append(model_variant)
    model_results['Negative'].append(negative_sampling_variant)
    model_results['Vector_size'].append(vector_size)
    model_results['Window'].append(window)
    model_results['Min_count'].append(min_count)
    model_results['Epochs'].append(epochs)
    model_results['Sample'].append(sample)
    model_results['Softmax'].append(hs_softmax)
    model_results['Word_pairs_test_Pearson_coeff'].append(pearson_coeff_word_pairs_eval)
    model_results['Word_pairs_test_Pearson_p-val'].append(pearson_p_val_word_pairs_eval)
    model_results['Word_pairs_test_Spearman_coeff'].append(spearman_coeff_word_pairs_eval)
    model_results['Word_pairs_test_Spearman_p-val'].append(spearman_p_val_word_pairs_eval)
    model_results['Word_pairs_test_Out-of-vocab_ratio'].append(out_of_vocab_ratio)
    model_results['Analogies_test'].append(analogies_eval)
    return model_results
