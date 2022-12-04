import simpful as sf
from simpful import FuzzySystem, FuzzySet, Triangular_MF, LinguisticVariable, Trapezoidal_MF


def weighted_rules_takagi_sugeno():
    # A simple fuzzy inference system for the tipping problem
    # Create a fuzzy system object
    FS = sf.FuzzySystem()

    # Define fuzzy sets and linguistic variables
    S_1 = sf.FuzzySet(points=[[0., 1.],  [5., 0.]], term="poor")
    S_2 = sf.FuzzySet(points=[[0., 0.], [5., 1.], [10., 0.]], term="good")
    S_3 = sf.FuzzySet(points=[[5., 0.],  [10., 1.]], term="excellent")
    FS.add_linguistic_variable("Service", sf.LinguisticVariable([S_1, S_2, S_3], concept="Service quality"))

    LV = sf.AutoTriangle(2, terms=["rancid", "delicious"], universe_of_discourse=[0,10], verbose=False)
    FS.add_linguistic_variable("Food", LV)

    # Define output crisp values
    FS.set_crisp_output_value("small", 5)
    FS.set_crisp_output_value("average", 15)

    # Define function for generous tip (food score + service score + 5%)
    FS.set_output_function("generous", "Food+Service+5")

    # Define fuzzy rules
    R1 = "IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS small) WEIGHT 0.2"
    R2 = "IF (Service IS good) THEN (Tip IS average) WEIGHT 1.0"
    R3 = "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous) WEIGHT 0.8"
    FS.add_rules([R1, R2, R3])

    # Set antecedents values
    FS.set_variable("Service", 4)
    FS.set_variable("Food", 8)

    # Perform Sugeno inference and print output
    print(FS.Sugeno_inference(["Tip"]))


def inference_simple_mamdani():
    # A simple fuzzy inference system for the tipping problem
    # Create a fuzzy system object
    FS = FuzzySystem()

    # Define fuzzy sets and linguistic variables
    S_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=5), term="poor")
    S_2 = FuzzySet(function=Triangular_MF(a=0, b=5, c=10), term="good")
    S_3 = FuzzySet(function=Triangular_MF(a=5, b=10, c=10), term="excellent")
    FS.add_linguistic_variable("Service", LinguisticVariable([S_1, S_2, S_3], concept="Service quality",
                                                             universe_of_discourse=[0, 10]))

    F_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="rancid")
    F_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=10), term="delicious")
    FS.add_linguistic_variable("Food",
                               LinguisticVariable([F_1, F_2], concept="Food quality", universe_of_discourse=[0, 10]))

    # Define output fuzzy sets and linguistic variable
    T_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="small")
    T_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=20), term="average")
    T_3 = FuzzySet(function=Trapezoidal_MF(a=10, b=20, c=25, d=25), term="generous")
    FS.add_linguistic_variable("Tip", LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0, 25]))

    # Define fuzzy rules
    R1 = "IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS small)"
    R2 = "IF (Service IS good) THEN (Tip IS average)"
    R3 = "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous)"
    FS.add_rules([R1, R2, R3])

    # Set antecedents values
    FS.set_variable("Service", 4)
    FS.set_variable("Food", 8)

    # Perform Mamdani inference and print output
    print(FS.Mamdani_inference(["Tip"]))


# TODO: Plot functions to the text of the thesis
def example_fuzzy_sets():
    import simpful as sf

    # A showcase of available fuzzy sets.

    # Crisp
    C_1 = sf.CrispSet(a=0, b=5, term="low")
    C_2 = sf.CrispSet(a=5, b=10, term="high")
    sf.LinguisticVariable([C_1, C_2], universe_of_discourse=[0, 10]).plot()

    # Point-based polygon
    P_1 = sf.FuzzySet(points=[[2.0, 1.0], [4.0, 0.25], [6.0, 0.0]], term="low")
    P_2 = sf.FuzzySet(points=[[2.0, 0.0], [4.0, 0.25], [6.0, 1.0]], term="high")
    sf.LinguisticVariable([P_1, P_2], universe_of_discourse=[0, 10]).plot()

    # Triangle
    Tri_1 = sf.TriangleFuzzySet(a=0, b=0, c=5, term="low")
    Tri_2 = sf.TriangleFuzzySet(a=0, b=5, c=10, term="medium")
    Tri_3 = sf.TriangleFuzzySet(a=5, b=10, c=10, term="high")
    sf.LinguisticVariable([Tri_1, Tri_2, Tri_3], universe_of_discourse=[0, 10]).plot()

    # Trapezoid
    Tra_1 = sf.TrapezoidFuzzySet(a=0, b=0, c=2, d=4, term="low")
    Tra_2 = sf.TrapezoidFuzzySet(a=2, b=4, c=6, d=8, term="medium")
    Tra_3 = sf.TrapezoidFuzzySet(a=6, b=8, c=10, d=10, term="high")
    sf.LinguisticVariable([Tra_1, Tra_2, Tra_3], universe_of_discourse=[0, 10]).plot()

    # Gaussian
    G_1 = sf.GaussianFuzzySet(mu=5, sigma=2, term="medium")
    G_2 = sf.InvGaussianFuzzySet(mu=5, sigma=2, term="not medium")
    sf.LinguisticVariable([G_1, G_2], universe_of_discourse=[0, 10]).plot()

    # Double Gaussian
    DG_1 = sf.DoubleGaussianFuzzySet(mu1=1, sigma1=0.1, mu2=1, sigma2=1, term="low")
    DG_2 = sf.DoubleGaussianFuzzySet(mu1=3.5, sigma1=1, mu2=6, sigma2=5, term="high")
    sf.LinguisticVariable([DG_1, DG_2], universe_of_discourse=[0, 10]).plot()

    # Sigmoid
    S_1 = sf.InvSigmoidFuzzySet(c=5, a=2, term="low")
    S_2 = sf.SigmoidFuzzySet(c=5, a=2, term="high")
    sf.LinguisticVariable([S_1, S_2], universe_of_discourse=[0, 10]).plot()

    # Function-based fuzzy set
    import numpy as np
    def fun1(x):
        return 0.5 * np.cos(0.314 * x) + 0.5

    def fun2(x):
        return 0.5 * np.sin(0.314 * x - 1.5) + 0.5

    F_1 = sf.FuzzySet(function=fun1, term="low")
    F_2 = sf.FuzzySet(function=fun2, term="high")
    sf.LinguisticVariable([F_1, F_2], universe_of_discourse=[0, 10]).plot()

    # Singletons set
    Ss_1 = sf.SingletonsSet(pairs=[[1.0, 0.2], [2.0, 0.8], [3.0, 0.4]], term="low")
    Ss_2 = sf.SingletonsSet(pairs=[[3.0, 0.3], [5.0, 0.9], [6.0, 0.1]], term="high")
    sf.LinguisticVariable([Ss_1, Ss_2], universe_of_discourse=[0, 10]).plot()


def example_output_space():
    import matplotlib.pylab as plt
    from numpy import linspace, array

    FS = FuzzySystem()

    S_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=5), term="poor")
    S_2 = FuzzySet(function=Triangular_MF(a=0, b=5, c=10), term="good")
    S_3 = FuzzySet(function=Triangular_MF(a=5, b=10, c=10), term="excellent")
    FS.add_linguistic_variable("Service", LinguisticVariable([S_1, S_2, S_3], concept="Service quality",
                                                             universe_of_discourse=[0, 10]))

    F_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="rancid")
    F_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=10), term="delicious")
    FS.add_linguistic_variable("Food",
                               LinguisticVariable([F_1, F_2], concept="Food quality", universe_of_discourse=[0, 10]))

    T_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="small")
    T_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=20), term="average")
    T_3 = FuzzySet(function=Trapezoidal_MF(a=10, b=20, c=25, d=25), term="generous")
    FS.add_linguistic_variable("Tip", LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0, 25]))

    R1 = "IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS small)"
    R2 = "IF (Service IS good) THEN (Tip IS average)"
    R3 = "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous)"
    FS.add_rules([R1, R2, R3])

    # Plotting surface
    xs = []
    ys = []
    zs = []
    DIVs = 20
    for x in linspace(0, 10, DIVs):
        for y in linspace(0, 10, DIVs):
            FS.set_variable("Food", x)
            FS.set_variable("Service", y)
            tip = FS.inference()['Tip']
            xs.append(x)
            ys.append(y)
            zs.append(tip)
    xs = array(xs)
    ys = array(ys)
    zs = array(zs)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xx, yy = plt.meshgrid(xs, ys)

    ax.plot_trisurf(xs, ys, zs, vmin=0, vmax=25, cmap='gnuplot2')
    ax.set_xlabel("Food")
    ax.set_ylabel("Service")
    ax.set_zlabel("Tip")
    ax.set_title("Simpful", pad=20)
    ax.set_zlim(0, 25)
    plt.tight_layout()
    plt.show()