from simpful import *


def inference_simple_mamdani_boosting_coeff(similarity, freshness):
    # A simple fuzzy inference system for the boostingping problem
    # Create a fuzzy system object
    FS = FuzzySystem()

    # Define fuzzy sets and linguistic variables
    S_1 = FuzzySet(function=Trapezoidal_MF(a=0, b=0.0, c=0.2, d=0.4), term="very_low")
    S_2 = FuzzySet(function=Trapezoidal_MF(a=0.2, b=0.3, c=0.4, d=0.45), term="low")
    S_3 = FuzzySet(function=Trapezoidal_MF(a=0.4, b=0.45, c=0.55, d=0.6), term="med")
    S_4 = FuzzySet(function=Trapezoidal_MF(a=0.7, b=0.75, c=0.8, d=0.9), term="med")
    S_5 = FuzzySet(function=Trapezoidal_MF(a=0.8, b=0.9, c=1, d=1), term="med")
    S_6 = FuzzySet(function=Trapezoidal_MF(a=0.55, b=0.6, c=0.7, d=0.75), term="med")
    FS.add_linguistic_variable("similarity", LinguisticVariable([S_1, S_2, S_3, S_4, S_5, S_6], concept="similarity Measure",
                                                             universe_of_discourse=[0.0, 1.0]))
    F_1 = FuzzySet(function=Trapezoidal_MF(a=0, b=0, c=24, d=48), term="old")
    F_2 = FuzzySet(function=Trapezoidal_MF(a=24, b=48, c=72, d=96), term="slightly_old")
    F_3 = FuzzySet(function=Trapezoidal_MF(a=72, b=96, c=120, d=96), term="current")
    F_4 = FuzzySet(function=Trapezoidal_MF(a=120, b=144, c=168, d=96), term="fresh")
    F_5 = FuzzySet(function=Trapezoidal_MF(a=168, b=192, c=336, d=336), term="very_fresh")
    FS.add_linguistic_variable("freshness",
                               LinguisticVariable([F_1, F_2, F_3, F_4, F_5], concept="freshness Measure", universe_of_discourse=[0, 336]))

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
    R3 = "IF (similarity IS very_high) OR (freshness IS old) THEN (boosting IS med)"
    FS.add_rules([R1, R2, R3])

    # Set antecedents values
    FS.set_variable("similarity", similarity)
    FS.set_variable("freshness", freshness)

    # Perform Mamdani inference and print output
    mamdani_inference = FS.Mamdani_inference(["boosting"])

    return mamdani_inference['boosting']
