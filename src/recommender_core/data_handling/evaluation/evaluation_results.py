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


def append_training_results(model_results, stats):
    model_results['Validation_Set'].append(stats['source'] + " " + stats['corpus_title'])
    model_results['Model_Variant'].append(stats['model_variant'])
    model_results['Negative'].append(stats['negative_sampling_variant'])
    model_results['Vector_size'].append(stats['vector_size'])
    model_results['Window'].append(stats['window'])
    model_results['Min_count'].append(stats['min_count'])
    model_results['Epochs'].append(stats['epochs'])
    model_results['Sample'].append(stats['sample'])
    model_results['Softmax'].append(stats['hs_softmax'])
    model_results['Word_pairs_test_Pearson_coeff'].append(stats['pearson_coeff_word_pairs_eval'])
    model_results['Word_pairs_test_Pearson_p-val'].append(stats['pearson_p_val_word_pairs_eval'])
    model_results['Word_pairs_test_Spearman_coeff'].append(stats['spearman_coeff_word_pairs_eval'])
    model_results['Word_pairs_test_Spearman_p-val'].append(stats['spearman_p_val_word_pairs_eval'])
    model_results['Word_pairs_test_Out-of-vocab_ratio'].append(stats['out_of_vocab_ratio'])
    model_results['Analogies_test'].append(stats['analogies_eval'])

    return model_results
