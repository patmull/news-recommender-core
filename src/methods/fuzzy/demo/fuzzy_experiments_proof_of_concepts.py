import logging

from simpful import FuzzySystem, FuzzySet, Triangular_MF, LinguisticVariable, Trapezoidal_MF

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from hybrid_methods.")

DEFAULT_CONCEPT = "Similarity Measure"


def inference_simple_mamdani_boosting_coeff(similarity, freshness):
    """
    # A simple fuzzy inference system for the Boostingping problem
    # Create a fuzzy system object
    """
    fs = FuzzySystem()

    # Define fuzzy sets and linguistic variables
    s_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="poor")
    s_2 = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1.0), term="good")
    s_3 = FuzzySet(function=Triangular_MF(a=0.5, b=1.0, c=1.0), term="excellent")
    fs.add_linguistic_variable("Similarity", LinguisticVariable([s_1, s_2, s_3],
                                                                concept=DEFAULT_CONCEPT,
                                                                universe_of_discourse=[0, 10]))

    f_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=100), term="fresh")
    f_2 = FuzzySet(function=Triangular_MF(a=0, b=100, c=100), term="old")
    fs.add_linguistic_variable("Freshness",
                               LinguisticVariable([f_1, f_2], concept="Freshness Measure",
                                                  universe_of_discourse=[0, 10]))

    # Define output fuzzy sets and linguistic variable
    t_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=5), term="small")
    t_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=20), term="average")
    t_3 = FuzzySet(function=Trapezoidal_MF(a=10, b=5, c=20, d=20), term="great")
    fs.add_linguistic_variable("Boosting", LinguisticVariable([t_1, t_2, t_3], universe_of_discourse=[0, 20]))

    # Define fuzzy rules
    r1 = "IF (Similarity IS poor) OR (Freshness IS fresh) THEN (Boosting IS small)"
    r2 = "IF (Similarity IS good) THEN (Boosting IS average)"
    r3 = "IF (Similarity IS excellent) OR (Freshness IS old) THEN (Boosting IS great)"
    fs.add_rules([r1, r2, r3])

    # Set antecedents values
    fs.set_variable("Similarity", similarity)
    fs.set_variable("Freshness", freshness)

    # Perform Mamdani inference and print output
    mamdani_inference = fs.Mamdani_inference(["Boosting"])

    return mamdani_inference['Boosting']


def fuzzy_weights_coeff():
    """
    # Demo of the Takagi-Sugeno inference using the system's similarity, freshness and boosting measurement.
    # Create a fuzzy system object
    @return: double of the Takagi-Sugeno fuzzy inference
    """
    import simpful as sf

    # A simple fuzzy inference system for the tipping problem
    # Create a fuzzy system object
    fs = sf.FuzzySystem()

    # Define fuzzy sets and linguistic variables
    s_1 = sf.FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="poor")
    s_2 = sf.FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1.0), term="good")
    s_3 = sf.FuzzySet(function=Triangular_MF(a=0.5, b=1.0, c=1.0), term="excellent")
    fs.add_linguistic_variable("Similarity", LinguisticVariable([s_1, s_2, s_3], concept=DEFAULT_CONCEPT,
                                                                universe_of_discourse=[0, 10]))

    lv = sf.AutoTriangle(2, terms=["old", "fresh"], universe_of_discourse=[0, 10], verbose=False)
    fs.add_linguistic_variable("Freshness", lv)

    # Define output crisp values
    fs.set_crisp_output_value("small", 5)
    fs.set_crisp_output_value("average", 15)

    # Define function for generous tip (food score + service score + 5%)
    fs.set_output_function("great", "Freshness+Similarity+5")

    # Define fuzzy rules
    r1 = "IF (Similarity IS poor) OR (Freshness IS old) THEN (Boosting IS small) WEIGHT 0.2"
    r2 = "IF (Similarity IS good) THEN (Boosting IS average) WEIGHT 1.0"
    r3 = "IF (Similarity IS excellent) OR (Freshness IS fresh) THEN (Boosting IS great) WEIGHT 0.8"
    fs.add_rules([r1, r2, r3])

    # Set antecedents values
    fs.set_variable("Similarity", 0.6)
    fs.set_variable("Freshness", 50)

    # Perform Sugeno inference and print output
    sugeno_inference = fs.Sugeno_inference(["Boosting"])

    return sugeno_inference


def inference_simple_mamdani_ensembling_ratio(similarity, freshness, returned_method):
    """
    Demo of the Mamdani inference using the system's similarity, freshness and boosting measurement.
    Create a fuzzy system object
    This method was later revisited and integrated into the Recommender Core module of the system.
    @return: double of the Takagi-Sugeno fuzzy inference
    """

    allowed_methods = ['terms_frequencies', 'word2vec', 'doc2vec']

    logging.debug("=================")
    logging.debug("FUZZY MODULE")
    logging.debug("=======================")
    logging.debug("Provided arguments:")
    logging.debug("------------------------")
    logging.debug("Similarity:")
    logging.debug(similarity)
    logging.debug("Freshness:")
    logging.debug(freshness)

    if returned_method not in allowed_methods:
        raise ValueError("Neither from passed returned method is in allowed methods")

    # A simple fuzzy inference system for the EnsembleRatio problem
    # Create a fuzzy system object
    FS = FuzzySystem()

    # Define fuzzy sets and linguistic variables
    S_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="poor")
    S_2 = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1.0), term="good")
    S_3 = FuzzySet(function=Triangular_MF(a=0.5, b=1.0, c=1.0), term="excellent")
    FS.add_linguistic_variable("Similarity", LinguisticVariable([S_1, S_2, S_3], concept=DEFAULT_CONCEPT,
                                                                universe_of_discourse=[0, 10]))

    F_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="fresh")
    F_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=10), term="old")
    FS.add_linguistic_variable("Freshness",
                               LinguisticVariable([F_1, F_2], concept="Freshness of Algorithm",
                                                  universe_of_discourse=[0, 10]))

    # Define output fuzzy sets and linguistic variable
    T_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="small")
    T_2 = FuzzySet(function=Triangular_MF(a=0, b=5, c=7), term="average")
    T_3 = FuzzySet(function=Trapezoidal_MF(a=7, b=9, c=10, d=10), term="great")
    FS.add_linguistic_variable("EnsembleRatioTfIdf", LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0, 10]))

    # Define output fuzzy sets and linguistic variable
    T_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="small")
    T_2 = FuzzySet(function=Triangular_MF(a=0, b=5, c=7), term="average")
    T_3 = FuzzySet(function=Trapezoidal_MF(a=7, b=9, c=10, d=10), term="great")
    FS.add_linguistic_variable("EnsembleRatioWord2Vec",
                               LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0, 10]))

    # Define output fuzzy sets and linguistic variable
    T_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="small")
    T_2 = FuzzySet(function=Triangular_MF(a=0, b=5, c=7), term="average")
    T_3 = FuzzySet(function=Trapezoidal_MF(a=7, b=9, c=10, d=10), term="great")
    FS.add_linguistic_variable("EnsembleRatioDoc2Vec",
                               LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0, 10]))

    # Define fuzzy rules
    R1 = "IF (Similarity IS poor) OR (Freshness IS fresh) THEN (EnsembleRatioTfIdf IS small)"
    R2 = "IF (Similarity IS good) THEN (EnsembleRatioTfIdf IS average)"
    R3 = "IF (Similarity IS excellent) OR (Freshness IS old) THEN (EnsembleRatioTfIdf IS great)"
    FS.add_rules([R1, R2, R3])

    # Define fuzzy rules
    R4 = "IF (Similarity IS poor) OR (Freshness IS fresh) THEN (EnsembleRatioWord2Vec IS small)"
    R5 = "IF (Similarity IS good) THEN (EnsembleRatioWord2Vec IS average)"
    R6 = "IF (Similarity IS excellent) OR (Freshness IS old) THEN (EnsembleRatioWord2Vec IS great)"
    FS.add_rules([R4, R5, R6])

    # Define fuzzy rules
    R4 = "IF (Similarity IS poor) OR (Freshness IS fresh) THEN (EnsembleRatioDoc2Vec IS small)"
    R5 = "IF (Similarity IS good) THEN (EnsembleRatioDoc2Vec IS average)"
    R6 = "IF (Similarity IS excellent) OR (Freshness IS old) THEN (EnsembleRatioDoc2Vec IS great)"
    FS.add_rules([R4, R5, R6])

    # Set antecedents values
    FS.set_variable("Similarity", similarity)
    FS.set_variable("Freshness", freshness)

    mamdani_inference_tfidf = FS.Mamdani_inference(["EnsembleRatioTfIdf"])
    mamdani_inference_word2vec = FS.Mamdani_inference(["EnsembleRatioWord2Vec"])
    mamdani_inference_doc2vec = FS.Mamdani_inference(["EnsembleRatioDoc2Vec"])

    if returned_method == 'terms_frequencies':
        return mamdani_inference_tfidf['EnsembleRatioTfIdf']

    if returned_method == 'word2vec':
        return mamdani_inference_word2vec['EnsembleRatioWord2Vec']

    if returned_method == 'doc2vec':
        return mamdani_inference_doc2vec['EnsembleRatioDoc2Vec']
