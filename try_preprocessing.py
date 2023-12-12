from src.prefillers.preprocessing.cz_preprocessing import preprocess
from src.recommender_core.data_handling.data_queries import preprocess_single_post_find_by_slug

print(preprocess_single_post_find_by_slug('jagr-ziskal-evropskou-cenu-za-celozivotni-prinos-porazil-hubla-ci-dacjuka')
      ['excerpt'].values[0]
      )

input_to_preprocess = input("Preprocessing input:")
preprocessing_output = preprocess(input_to_preprocess)
print("preprocessing_output: \n", preprocessing_output)
