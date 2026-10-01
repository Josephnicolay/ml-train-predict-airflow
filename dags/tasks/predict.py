import os
from airflow.decorators import task

@task.virtualenv(
    task_id="preprocess_data",
    multiple_outputs=True,
    requirements=["numpy", "pandas", "scikit-learn"],
    pip_install_options=["--no-cache-dir"]
)
def preprocess_data():
    import os
    from pathlib import Path
    import numpy as np
    import pandas as pd

    AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
    DATA_DIR = Path(AIRFLOW_HOME) / "dags/data/"
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

    df = pd.read_fwf(DATA_DIR / "raw/data.txt", colspecs=colspecs, names=colsnames,
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

    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
    processed_data_path = DATA_DIR / "processed/data_processed.csv"
    df.to_csv(processed_data_path, index=False)

    return {
        "processed_data": str(processed_data_path)
    }

@task.virtualenv(
    task_id="predict",
    requirements=["cloudpickle", "pandas", "scikit-learn", "pytz"],
    pip_install_options=["--no-cache-dir"],
    multiple_outputs=True,
)
def predict(data_path, params=None):
    import os
    from pathlib import Path
    import cloudpickle
    import pandas as pd
    from datetime import datetime
    import pytz

    TZ = os.getenv("TZ")

    AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
    DATA_DIR = Path(AIRFLOW_HOME) / "dags/data"
    MODELS_DIR = Path(AIRFLOW_HOME) / "dags/model"
    
    with open(MODELS_DIR / params["model"], "rb") as f:
        model = cloudpickle.load(f)

    data = pd.read_csv(data_path)
    predictions = model.predict(data)

    predictions_path = DATA_DIR / "processed/predictions.csv"
    predictions_df = pd.DataFrame(predictions, columns=["MassExcessPrediction"])
    predictions_df["date"] = datetime.now(tz=pytz.timezone(TZ)).isoformat()
    predictions_df = pd.concat([data[["Z", "A", "El"]], predictions_df], axis=1)
    predictions_df.to_csv(predictions_path, index=False)

    print(f"Predictions based on {data_path} saved to {predictions_path}")
    return {
        "predictions": str(predictions_path),
        "numberOfPredictions": len(predictions)
    }