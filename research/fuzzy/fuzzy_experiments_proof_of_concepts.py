import simpful as sf
from simpful import FuzzySystem, FuzzySet, Triangular_MF, LinguisticVariable, Trapezoidal_MF


def inference_simple_mamdani_boosting_coeff():
    # A simple fuzzy inference system for the Boostingping problem
    # Create a fuzzy system object
    FS = FuzzySystem()

    # Define fuzzy sets and linguistic variables
    S_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="poor")
    S_2 = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1.0), term="good")
    S_3 = FuzzySet(function=Triangular_MF(a=0.5, b=1.0, c=1.0), term="excellent")
    FS.add_linguistic_variable("Similarity", LinguisticVariable([S_1, S_2, S_3], concept="Similarity Measure",
                                                             universe_of_discourse=[0, 10]))

    F_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=100), term="fresh")
    F_2 = FuzzySet(function=Triangular_MF(a=0, b=100, c=100), term="old")
    FS.add_linguistic_variable("Freshness",
                               LinguisticVariable([F_1, F_2], concept="Freshness Measure", universe_of_discourse=[0, 10]))

    # Define output fuzzy sets and linguistic variable
    T_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=5), term="small")
    T_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=20), term="average")
    T_3 = FuzzySet(function=Trapezoidal_MF(a=10, b=5, c=20, d=20), term="great")
    FS.add_linguistic_variable("Boosting", LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0, 20]))

    # Define fuzzy rules
    # TODO: This will be loaded from Redis
    R1 = "IF (Similarity IS poor) OR (Freshness IS fresh) THEN (Boosting IS small)"
    # TODO: This will be loaded from Redis
    R2 = "IF (Similarity IS good) THEN (Boosting IS average)"
    # TODO: This will be loaded from Redis
    R3 = "IF (Similarity IS excellent) OR (Freshness IS old) THEN (Boosting IS great)"
    FS.add_rules([R1, R2, R3])

    # Set antecedents values
    FS.set_variable("Similarity", 0.6)
    FS.set_variable("Freshness", 50)

    # Perform Mamdani inference and print output
    # TODO: This will got to Hybrid algorithm
    print(FS.Mamdani_inference(["Boosting"]))


def fuzzy_weights_coeff():
    import simpful as sf

    # A simple fuzzy inference system for the tipping problem
    # Create a fuzzy system object
    FS = sf.FuzzySystem()

    # Define fuzzy sets and linguistic variables
    S_1 = sf.FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="poor")
    S_2 = sf.FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1.0), term="good")
    S_3 = sf.FuzzySet(function=Triangular_MF(a=0.5, b=1.0, c=1.0), term="excellent")
    FS.add_linguistic_variable("Similarity", LinguisticVariable([S_1, S_2, S_3], concept="Similarity Measure",
                                                             universe_of_discourse=[0, 10]))

    LV = sf.AutoTriangle(2, terms=["old", "fresh"], universe_of_discourse=[0, 10], verbose=False)
    FS.add_linguistic_variable("Freshness", LV)

    # Define output crisp values
    FS.set_crisp_output_value("small", 5)
    FS.set_crisp_output_value("average", 15)

    # Define function for generous tip (food score + service score + 5%)
    FS.set_output_function("great", "Freshness+Similarity+5")

    # Define fuzzy rules
    R1 = "IF (Similarity IS poor) OR (Freshness IS old) THEN (Boosting IS small) WEIGHT 0.2"
    R2 = "IF (Similarity IS good) THEN (Boosting IS average) WEIGHT 1.0"
    R3 = "IF (Similarity IS excellent) OR (Freshness IS fresh) THEN (Boosting IS great) WEIGHT 0.8"
    FS.add_rules([R1, R2, R3])

    # Set antecedents values
    FS.set_variable("Similarity", 0.6)
    FS.set_variable("Freshness", 50)

    # Perform Sugeno inference and print output
    print(FS.Sugeno_inference(["Boosting"]))

def inference_simple_mamdani_ensembling_ratio():
    # A simple fuzzy inference system for the EnsembleRatio problem
    # Create a fuzzy system object
    FS = FuzzySystem()

    # Define fuzzy sets and linguistic variables
    S_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="poor")
    S_2 = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1.0), term="good")
    S_3 = FuzzySet(function=Triangular_MF(a=0.5, b=1.0, c=1.0), term="excellent")
    FS.add_linguistic_variable("Similarity", LinguisticVariable([S_1, S_2, S_3], concept="Similarity Measure",
                                                             universe_of_discourse=[0, 10]))

    F_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="fresh")
    F_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=10), term="old")
    FS.add_linguistic_variable("Precision",
                               LinguisticVariable([F_1, F_2], concept="Precision of Algorithm", universe_of_discourse=[0, 10]))

    # Define output fuzzy sets and linguistic variable
    T_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="small")
    T_2 = FuzzySet(function=Triangular_MF(a=0, b=5, c=7), term="average")
    T_3 = FuzzySet(function=Trapezoidal_MF(a=7, b=9, c=10, d=10), term="great")
    FS.add_linguistic_variable("EnsembleRatio", LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0, 10]))

    # Define fuzzy rules
    # TODO: This will be loaded from Redis
    R1 = "IF (Similarity IS poor) OR (Precision IS fresh) THEN (EnsembleRatio IS small)"
    # TODO: This will be loaded from Redis
    R2 = "IF (Similarity IS good) THEN (EnsembleRatio IS average)"
    # TODO: This will be loaded from Redis
    R3 = "IF (Similarity IS excellent) OR (Precision IS old) THEN (EnsembleRatio IS great)"
    FS.add_rules([R1, R2, R3])

    # Set antecedents values
    FS.set_variable("Similarity", 0.4)
    FS.set_variable("Precision", 4)

    # Perform Mamdani inference and print output
    # TODO: This will go to hybrid algorithm
    print(FS.Mamdani_inference(["EnsembleRatio"]))
