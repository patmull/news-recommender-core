import simpful as sf
from simpful import FuzzySystem, FuzzySet, Triangular_MF, LinguisticVariable, Trapezoidal_MF

DEFAULT_CONCEPT = "Service quality"


# noinspection PyPep8Naming
def weighted_rules_takagi_sugeno():
    """
    A simple fuzzy inference system for the tipping problem
    Create a fuzzy system object
    """
    fs = sf.FuzzySystem()

    # Define fuzzy sets and linguistic variables
    S_1 = sf.FuzzySet(points=[[0., 1.], [5., 0.]], term="poor")
    S_2 = sf.FuzzySet(points=[[0., 0.], [5., 1.], [10., 0.]], term="good")
    S_3 = sf.FuzzySet(points=[[5., 0.], [10., 1.]], term="excellent")
    fs.add_linguistic_variable("Service", sf.LinguisticVariable([S_1, S_2, S_3], concept=DEFAULT_CONCEPT))

    LV = sf.AutoTriangle(2, terms=["rancid", "delicious"], universe_of_discourse=[0, 10], verbose=False)
    fs.add_linguistic_variable("Food", LV)

    # Define output crisp values
    fs.set_crisp_output_value("small", 5)
    fs.set_crisp_output_value("average", 15)

    # Define function for generous tip (food score + service score + 5%)
    fs.set_output_function("generous", "Food+Service+5")

    # Define fuzzy rules
    R1 = "IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS small) WEIGHT 0.2"
    R2 = "IF (Service IS good) THEN (Tip IS average) WEIGHT 1.0"
    R3 = "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous) WEIGHT 0.8"
    fs.add_rules([R1, R2, R3])

    # Set antecedents values
    fs.set_variable("Service", 4)
    fs.set_variable("Food", 8)

    # Perform Sugeno inference and print output
    print(fs.Sugeno_inference(["Tip"]))


def create_fuzzy_set(points, term, membership_function):
    if membership_function == "triangular":
        return sf.FuzzySet(points=points, term=term)
    elif membership_function == "trapezoidal":
        return sf.FuzzySet(function=sf.Trapezoidal_MF(points[0][0], points[0][1], points[1][0], points[1][1]),
                           term=term)
    elif membership_function == "gaussian":
        return sf.FuzzySet(function=sf.GaussianFuzzySet(points[0][0], points[0][1], term=term))
    # Add other membership functions here
    else:
        raise ValueError("Invalid membership function type")


# noinspection PyPep8Naming
def inference_simple_mamdani():
    """
    # A simple fuzzy inference system for the tipping problem
    # Create a fuzzy system object
    """
    fs = FuzzySystem()

    # Define fuzzy sets and linguistic variables
    s_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=5), term="poor")
    S_2 = FuzzySet(function=Triangular_MF(a=0, b=5, c=10), term="good")
    S_3 = FuzzySet(function=Triangular_MF(a=5, b=10, c=10), term="excellent")
    fs.add_linguistic_variable("Service", LinguisticVariable([s_1, S_2, S_3], concept=DEFAULT_CONCEPT,
                                                             universe_of_discourse=[0, 10]))

    F_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="rancid")
    F_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=10), term="delicious")
    fs.add_linguistic_variable("Food",
                               LinguisticVariable([F_1, F_2], concept="Food quality", universe_of_discourse=[0, 10]))

    # Define output fuzzy sets and linguistic variable
    T_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="small")
    T_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=20), term="average")
    T_3 = FuzzySet(function=Trapezoidal_MF(a=10, b=20, c=25, d=25), term="generous")
    fs.add_linguistic_variable("Tip", LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0, 25]))

    # Define fuzzy rules
    R1 = "IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS small)"
    R2 = "IF (Service IS good) THEN (Tip IS average)"
    R3 = "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous)"
    fs.add_rules([R1, R2, R3])

    # Set antecedents values
    fs.set_variable("Service", 4)
    fs.set_variable("Food", 8)


# noinspection PyPep8Naming
def example_fuzzy_sets():
    """
    Example of evalutation of the Fuzzy sets in Simpful package
    @return:
    """
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
    tri_1 = sf.TriangleFuzzySet(a=0, b=0, c=5, term="low")
    tri_2 = sf.TriangleFuzzySet(a=0, b=5, c=10, term="medium")
    tri_3 = sf.TriangleFuzzySet(a=5, b=10, c=10, term="high")
    sf.LinguisticVariable([tri_1, tri_2, tri_3], universe_of_discourse=[0, 10]).plot()

    # Trapezoid
    tra_1 = sf.TrapezoidFuzzySet(a=0, b=0, c=2, d=4, term="low")
    tra_2 = sf.TrapezoidFuzzySet(a=2, b=4, c=6, d=8, term="medium")
    tra_3 = sf.TrapezoidFuzzySet(a=6, b=8, c=10, d=10, term="high")
    sf.LinguisticVariable([tra_1, tra_2, tra_3], universe_of_discourse=[0, 10]).plot()

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
    ss_1 = sf.SingletonsSet(pairs=[[1.0, 0.2], [2.0, 0.8], [3.0, 0.4]], term="low")
    ss_2 = sf.SingletonsSet(pairs=[[3.0, 0.3], [5.0, 0.9], [6.0, 0.1]], term="high")
    sf.LinguisticVariable([ss_1, ss_2], universe_of_discourse=[0, 10]).plot()


# noinspection PyPep8Naming
def example_output_space():
    """
    Example of use of the plotting with the Simpful package.
    @return:
    """
    import matplotlib.pylab as plt
    from numpy import linspace, array

    FS = FuzzySystem()

    S_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=5), term="poor")
    S_2 = FuzzySet(function=Triangular_MF(a=0, b=5, c=10), term="good")
    S_3 = FuzzySet(function=Triangular_MF(a=5, b=10, c=10), term="excellent")
    FS.add_linguistic_variable("Service", LinguisticVariable([S_1, S_2, S_3], concept=DEFAULT_CONCEPT,
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
    # noinspection PyPep8Naming
    R3 = "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous)"
    FS.add_rules([R1, R2, R3])

    # Plotting surface
    xs = []
    ys = []
    zs = []
    divs = 20
    for x in linspace(0, 10, divs):
        for y in linspace(0, 10, divs):
            FS.set_variable("Food", x)
            FS.set_variable("Service", y)
            tip = FS.inference()['Tip']
            xs.append(x)
            ys.append(y)
            zs.append(tip)
    xs = array(xs)
    ys = array(ys)
    zs = array(zs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(xs, ys, zs, vmin=0, vmax=25, cmap='gnuplot2')
    ax.set_xlabel("Food")
    ax.set_ylabel("Service")
    ax.set_zlabel("Tip")
    ax.set_title("Simpful", pad=20)
    ax.set_zlim(0, 25)
    plt.tight_layout()
    plt.show()
