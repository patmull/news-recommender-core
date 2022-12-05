import logging

from simpful import FuzzySystem, Trapezoidal_MF, FuzzySet, LinguisticVariable


def inference_mamdani_boosting_coeff(similarity, freshness):
    # A simple fuzzy inference system for the boostingping problem
    # Create a fuzzy system object
    FS = FuzzySystem()

    # Define fuzzy sets and linguistic variables
    S_1 = FuzzySet(function=Trapezoidal_MF(a=0, b=0.0, c=0.2, d=0.4), term="very_low")
    S_2 = FuzzySet(function=Trapezoidal_MF(a=0.2, b=0.3, c=0.4, d=0.45), term="low")
    S_3 = FuzzySet(function=Trapezoidal_MF(a=0.4, b.45, c=0.55, d=0.6), term="med")
    S_4 = FuzzySet(function=Trapezoidal_MF(a=0.7, b=0.75, c=0.8, d=0.9), term="high")
    S_5 = FuzzySet(function=Trapezoidal_MF(a=0.8, b=0.9, c=1, d=1), term="very_high")
    S_6 = FuzzySet(function=Trapezoidal_MF(a=0.55, b=0.6, c=0.7, d=0.75), term="medium_high")
    FS.add_linguistic_variable("similarity",
                               LinguisticVariable([S_1, S_2, S_3, S_4, S_5, S_6], concept="similarity Measure",
                                                  universe_of_discourse=[0.0, 1.0]))

    F_1 = FuzzySet(function=Trapezoidal_MF(a=0, b=0, c=24, d=48), term="old")
    F_2 = FuzzySet(function=Trapezoidal_MF(a=24, b=48, c=72, d=96), term="slightly_old")
    F_3 = FuzzySet(function=Trapezoidal_MF(a=72, b=96, c=120, d=96), term="current")
    F_4 = FuzzySet(function=Trapezoidal_MF(a=120, b=144, c=168, d=96), term="fresh")
    F_5 = FuzzySet(function=Trapezoidal_MF(a=168, b=192, c=336, d=336), term="very_fresh")
    FS.add_linguistic_variable("freshness",
                               LinguisticVariable([F_1, F_2, F_3, F_4, F_5], concept="freshness Measure",
                                                  universe_of_discourse=[0, 336]))

    # Define output fuzzy sets and linguistic variable
    T_1 = FuzzySet(function=Trapezoidal_MF(a=0, b=0, c=0.2, d=0.3), term="very_low")
    T_2 = FuzzySet(function=Trapezoidal_MF(a=0.2, b=0.3, c=0.4, d=0.45), term="low")
    T_3 = FuzzySet(function=Trapezoidal_MF(a=0.4, b=0.45, c=0.55, d=0.6), term="med")
    T_4 = FuzzySet(function=Trapezoidal_MF(a=0.55, b=0.6, c=0.7, d=0.8), term="high")
    T_5 = FuzzySet(function=Trapezoidal_MF(a=0.7, b=0.8, c=1, d=1), term="very_high")
    FS.add_linguistic_variable("boosting", LinguisticVariable([T_1, T_2, T_3, T_4, T_5],
                                                              universe_of_discourse=[0, 1.0]))

    # Define fuzzy rules
    R1 = "IF (similarity IS very_high) AND ((freshness IS very_fresh) OR (freshness IS fresh)) THEN (boosting IS very_high)"
    R2 = "IF (similarity IS very_high) AND ((freshness IS current) OR (freshness IS slightly_old)) THEN (boosting IS high)"
    R3 = "IF (similarity IS very_high) AND (freshness IS old) THEN (boosting IS med)"
    R4 = "IF (similarity IS high) AND ((freshness IS very_fresh) OR (freshness IS fresh)) THEN (boosting IS high)"
    R5 = "IF (similarity IS high) AND (freshness IS slightly_old) THEN (boosting IS very_high)"
    R6 = "IF (similarity IS high) AND (freshness IS current) THEN (boosting IS med)"
    R7 = "IF (similarity IS high) AND (freshness IS old) THEN (boosting IS high)"
    R8 = "IF (similarity IS medium_high) AND ((freshness IS fresh) OR (freshness IS current)) THEN (boosting IS high)"
    R9 = "IF (similarity IS medium_high) AND ((freshness IS slightly_old) OR (freshness IS old)) THEN (boosting IS med)"
    R10 = "IF (similarity IS medium_high) AND (freshness IS very_fresh) THEN (boosting IS med)"
    R11 = "IF (similarity IS low) AND ((freshness IS old) OR (freshness IS slightly_old) OR " \
          "(freshness IS current) OR (freshness IS fresh) OR (freshness IS very_fresh)) THEN (boosting IS low)"
    R12 = "IF (similarity IS very_low) AND ((freshness IS old) OR (freshness IS slightly_old) OR (freshness IS current) " \
          "OR (freshness IS fresh)) THEN (boosting IS very_low)"
    R13 = "IF (similarity IS very_low) AND ((freshness IS very_fresh) OR (freshness IS fresh)) THEN (boosting IS very_low)"
    R14 = "IF (similarity IS med) AND (freshness IS very_fresh) THEN (boosting IS high)"
    R15 = "IF (similarity IS med) AND ((freshness IS old) OR (freshness IS slightly_old) OR (freshness IS current)) THEN (boosting IS med)"
    R16 = "IF (similarity IS med) AND (freshness IS fresh) THEN (boosting IS med)"

    FS.add_rules([R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16])

    # Set antecedents values
    FS.set_variable("similarity", similarity)
    FS.set_variable("freshness", freshness)

def inference_simple_mamdani_mixi(similarity, freshness, returned_method):
    allowed_methods = ['tfidf', 'word2vec', 'doc2vec']

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
    S_1 = FuzzySet(function=Trapezoidal_MF(a=0, b=0, c=0.2, d=0.4), term="high")
    S_2 = FuzzySet(function=Trapezoidal_MF(a=0, b=0.5, c=1.0), term="medium")
    S_3 = FuzzySet(function=Trapezoidal_MF(a=0.5, b=1.0, c=1.0), term="excellent")
    FS.add_linguistic_variable("Similarity", LinguisticVariable([S_1, S_2, S_3], concept="Similarity Measure",
                                                                universe_of_discourse=[0, 10]))

    F_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="fresh"
    F_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=10), term="old")
    FS.add_linguistic_variable("Freshness",ng
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
    # Perform Mamdani inference and print output
    # TODO: This will go to hybrid algorithm
    print(mamdani_inference_word2vec)

    if returned_method == 'tfidf':
        return mamdani_inference_tfidf['EnsembleRatioTfIdf']

    if returned_method == 'word2vec':
        return mamdani_inference_word2vec['EnsembleRatioWord2Vec']

    if returned_method == 'doc2vec':
        return mamdani_inference_doc2vec['EnsembleRatioDoc2Vec']
