import os
from airflow.decorators import task


@task.virtualenv(
    task_id="get_data",
    requirements=["requests"],
    pip_install_options=["--no-cache-dir"]
)
def get_data():
    import os
    from pathlib import Path
    import requests

    AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
    DATA_DIR = Path(AIRFLOW_HOME) / "dags/data/raw"

    url = "https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt"
    response = requests.get(url)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_path = DATA_DIR / "data.txt"
    with open(data_path, "wb") as f:
        f.write(response.content)
    return str(data_path)

@task.virtualenv(
    task_id="preprocess_data",
    multiple_outputs=True,
    requirements=["numpy", "pandas", "scikit-learn"],
    pip_install_options=["--no-cache-dir"]
)
def preprocess_data(data_path):
    import os
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
    DATA_DIR = Path(AIRFLOW_HOME) / "dags/data/processed"
    HEADER_LINES = 36
    _MAGIC_NUMBERS = [0, 2, 8, 20, 28, 50, 82, 126, 184, np.inf]

    def to_num(s):
            """Return (value, is_estimated). '*' or blank -> NaN."""
            s = s.strip()
            if s in ("", "*"):
                return np.nan, False
            est = "#" in s
            try:
                return float(s.replace("#", ".")), est
            except ValueError:
                print(f"Warning: could not convert '{s}' to float.")
                return np.nan, est

    def delta_nucleons(nucleon_number)-> int:
        """
        Calculate the number of outer shell nucleons for a given nucleon number.

        Args:
            nucleon_number (int): the number of nucleons (either protons or neutrons).

        Returns:
            int: The number of nucleons outter or missing from the closest shell.
        
        Notes:
            Magic numbers 126 (protons) and 184 (neutrons) are theoretical and 
            have not been experimentally confirmed.
            For real nuclei, 295 Og is the heaviest known isotope, with Z=118 and N=177.
            For superheavy nuclei, results beyond Z=82 or N=126 depend on these predictions.
        """
        assert isinstance(nucleon_number, int) and nucleon_number >= 0, "nucleon_number must be a non-negative integer"

        lower = max([m for m in _MAGIC_NUMBERS if m <= nucleon_number])
        upper = min([m for m in _MAGIC_NUMBERS if m > nucleon_number])
        return int(min(nucleon_number - lower, upper - nucleon_number))

    def even_odd(n):
        """
        Determine if a number is even or odd.

        Args:
            n (int): The number to check.

        Returns:
            int: 0 if the number is even, 1 if the number is odd.
        """
        return n % 2

    colspecs = [(1,4),(4,9),(9,14),(14,19),(20,23),(23,27),
            (28,42),(42,54),(54,67),(68,78),
            (79,81),(81,94),(94,105),
            (106,109),(110,123),(123,135)]
    colsnames = ["N_Z","N","Z","A","El","Orig",
            "MassExcess","MassExcess_unc",
            "BE_per_A","BE_per_A_unc",
            "BetaType","BetaDecay","BetaDecay_unc",
            "AM_int","AM_frac","AM_unc"]
    
    num_cols = ["MassExcess","MassExcess_unc","BE_per_A","BE_per_A_unc",
                "BetaDecay","BetaDecay_unc","AM_frac","AM_unc"]

    df = pd.read_fwf(data_path, colspecs=colspecs, names=colsnames,
                    skiprows=HEADER_LINES, dtype=str, keep_default_na=False)

    for c in num_cols:
        vals = df[c].map(to_num)
        if vals.isnull().all():
            print(f"Warning: all values in column '{c}' could not be converted to numeric.")
            os._exit(1)
        df[c] = vals.map(lambda t: t[0])
        df[c + "_est"] = vals.map(lambda t: t[1])

    for c in ["N_Z","N","Z","A","AM_int"]:
        df[c] = pd.to_numeric(df[c].str.strip(), errors="coerce")

    # full atomic mass in micro-u
    df["AtomicMass_uu"] = df["AM_int"] * 1e6 + df["AM_frac"]
    df["El"] = df["El"].str.strip()
    df = df.drop(columns=["AM_int","AM_frac"])

    df['DeltaZ'] = df['Z'].apply(delta_nucleons)
    df['DeltaN'] = df['N'].apply(delta_nucleons)
    df['DeltaA'] = df['DeltaZ'] + df['DeltaN']
    df['EvenOddZ'] = df['Z'].apply(even_odd)
    df['EvenOddN'] = df['N'].apply(even_odd)
        
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_data_path = DATA_DIR / "data.csv"
    train_data_path = DATA_DIR / "train_data.csv"
    test_data_path = DATA_DIR / "test_data.csv"

    df.to_csv(processed_data_path, index=False)
    train_data.to_csv(train_data_path, index=False)
    test_data.to_csv(test_data_path, index=False)

    return {
        "data": str(processed_data_path),
        "train": str(train_data_path),
        "test": str(test_data_path)
    }

@task.virtualenv(
    task_id="train_model",
    requirements=["cloudpickle", "pandas", "scikit-learn"],
    pip_install_options=["--no-cache-dir"],
    multiple_outputs=True,
)
def train_model(train_data_path):    
    import os
    from pathlib import Path
    import cloudpickle
    import pandas as pd
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, PowerTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.metrics import root_mean_squared_error
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import RandomForestRegressor

    AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
    WORKING_DIR = Path(AIRFLOW_HOME) / "dags/"

    FEATURES = ['N_Z', 'N', 'Z', 'A', 'BE_per_A', 'DeltaZ', 'DeltaN',
                'DeltaA', 'EvenOddZ', 'EvenOddN']
    LABEL = ['MassExcess']
    _MAGIC_NUMBERS = [0, 2, 8, 20, 28, 50, 82, 126, 184]

    class ClusterSimilarity(BaseEstimator, TransformerMixin):
        def __init__(self, gamma=1.0, random_state=None):
            self.cluster_centers_ = list(zip(_MAGIC_NUMBERS, _MAGIC_NUMBERS))
            self.gamma = gamma
            self.random_state = random_state

        def fit(self, X, y=None):
            # No fitting necessary for this transformer
            return self  # always return self!

        def transform(self, X):
            return rbf_kernel(X, self.cluster_centers_, gamma=self.gamma)

        def get_feature_names_out(self, names=None):
            return [f"boundary_{i}" for i in range(len(self.cluster_centers_))]

    def surface_term(A):
        return A**(2/3)

    def surface_term_pipeline():
        return make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0),
            FunctionTransformer(
                surface_term, 
                feature_names_out=(
                    lambda transformer, input_features: ["surface"]
                )
            ),
            MinMaxScaler()
        )

    def coulomb_term(X):
        Z = X[:, [0]]
        A = X[:, [1]]
        return Z*(Z-1)/A**(1/3)

    def coulomb_term_pipeline():
        return make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0),
            FunctionTransformer(
                coulomb_term, 
                feature_names_out=(
                    lambda transformer, input_features: ["coulomb"]
                )
            ),
            MinMaxScaler()
        )

    def asymmetry_term(X):
        N = X[:, [0]]
        Z = X[:, [1]]
        return (N-Z)**2/(N+Z)

    def asymmetry_term_pipeline():
        return make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0),
            FunctionTransformer(
                asymmetry_term, 
                feature_names_out=(
                    lambda transformer, input_features: ["asymmetry"]
                )
            ),
            MinMaxScaler()
        )
    num_pipeline = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        MinMaxScaler()
    )

    preprocessing = ColumnTransformer(
        [
            ("num", num_pipeline, ["N", "Z", "A", "DeltaN", "DeltaZ", "DeltaA", "BE_per_A"]),
            ("cat", "passthrough", ["EvenOddZ", "EvenOddN"]),
            ("coef1", surface_term_pipeline(), ["A"]),
            ("coef2", coulomb_term_pipeline(), ["Z", "A"]),
            ("coef3", asymmetry_term_pipeline(), ["N", "Z"]),
            ("sim", ClusterSimilarity(gamma=0.01), ["N", "Z"])
        ],
        remainder="drop",
    )

    data = pd.read_csv(train_data_path, usecols=FEATURES + LABEL)
    X = data[FEATURES]
    y = data[LABEL].to_numpy().ravel()
    reg = TransformedTargetRegressor(RandomForestRegressor(),
                                      transformer=PowerTransformer())
    model = make_pipeline(
        preprocessing,
        reg
    )
    model.fit(X, y)
    predictions = model.predict(X)
    rmse = root_mean_squared_error(y, predictions)
    print(f"Training RMSE: {rmse:.2f} [KeV]")

    (WORKING_DIR / "model").mkdir(parents=True, exist_ok=True)
    model_path = WORKING_DIR / f"model/model.pkl"
    with open(model_path, "wb") as f:
        cloudpickle.dump(model, f)

    return {
        "model": str(model_path),
        "rmse": rmse
    }

@task.virtualenv(
    task_id="evaluate_model",
    requirements=["cloudpickle", "pandas", "scikit-learn"],
    pip_install_options=["--no-cache-dir"],
    multiple_outputs=True,
)
def evaluate_model(model_path, test_data_path):
    import cloudpickle
    import pandas as pd
    from sklearn.metrics import root_mean_squared_error

    FEATURES = ['N_Z', 'N', 'Z', 'A', 'BE_per_A', 'DeltaZ', 'DeltaN',
                'DeltaA', 'EvenOddZ', 'EvenOddN']
    LABEL = ['MassExcess']

    with open(model_path, "rb") as f:
        model = cloudpickle.load(f)

    test_data = pd.read_csv(test_data_path)
    
    X_test = test_data[FEATURES]
    y_test = test_data[LABEL].to_numpy().ravel()

    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    print(f"Evaluation RMSE: {rmse:.2f} [KeV]")
    return {"rmse": rmse}
