import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CL Slag Cu-Loss Counterfactual",
    #page_icon="🔥",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_CLASS = 1

# Target probability for lower-Cu-loss class
DESIRED_PROB = 0.90

# Differential Evolution
MAXITER = 200
POPSIZE = 25

# Multiple independent optimization runs
SEEDS = [42, 43, 44]

# Strong penalty for failing to reach target probability
TARGET_PENALTY = 1_000_000.0

# Weight for process movement
CHANGE_WEIGHT = 1.0

# ============================================================
# IMPORTANT:
# Only these variables are allowed to change.
# ============================================================

CONTROL_VARIABLES = [

    "C-SLAG FEED RATE - S Furnace",

    "SILICA FEED RATE ",

    "S-FURNACE AIR",

    "S-FURNACE OXYGEN"

]


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.getcwd()

MODEL_PATH = os.path.join(
    BASE_DIR,
    "rf_final.pkl"
)

FEATURE_ORDER_PATH = os.path.join(
    BASE_DIR,
    "feature_order.pkl"
)

X_TRAIN_PATH = os.path.join(
    BASE_DIR,
    "X_train.npy"
)

Y_TRAIN_PATH = os.path.join(
    BASE_DIR,
    "y_train.npy"
)

RAW_X_TRAIN_PATH = os.path.join(
    BASE_DIR,
    "raw_X_train.pkl"
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    rf = joblib.load(
        MODEL_PATH
    )

    feature_order = joblib.load(
        FEATURE_ORDER_PATH
    )

    X_train = np.load(
        X_TRAIN_PATH
    )

    y_train = np.load(
        Y_TRAIN_PATH
    )

    raw_X_train = joblib.load(
        RAW_X_TRAIN_PATH
    )

    return (
        rf,
        feature_order,
        X_train,
        y_train,
        raw_X_train
    )


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [

    MODEL_PATH,
    FEATURE_ORDER_PATH,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    RAW_X_TRAIN_PATH

]


missing_files = [

    path

    for path in required_files

    if not os.path.exists(path)

]


if missing_files:

    st.error(
        "Required artifact files are missing."
    )

    for path in missing_files:

        st.write(
            f"- {os.path.basename(path)}"
        )

    st.stop()


# ============================================================
# LOAD
# ============================================================

try:

    (
        rf,
        FEATURE_ORDER,
        X_train,
        y_train,
        raw_X_train

    ) = load_artifacts()

except Exception as e:

    st.error(
        f"Error loading model artifacts:\n\n{e}"
    )

    st.stop()


# ============================================================
# FEATURE ORDER VERIFICATION
# ============================================================

EXPECTED_FEATURE_ORDER = [

    "Cu_Input",

    "Fe_Input",

    "Sulfur_Feed",

    "Matte_Flow",

    "Slag_Flow",

    "SILICA FEED RATE ",

    "O2_to_Sulfur",

    "Fe/SiO2",

    "Fe3O4_Cls",

    "Matte Grade"

]


if list(FEATURE_ORDER) != EXPECTED_FEATURE_ORDER:

    st.error(
        "Feature order in feature_order.pkl does not match "
        "the expected RF feature order."
    )

    st.write(
        "Loaded feature order:"
    )

    st.write(
        FEATURE_ORDER
    )

    st.write(
        "Expected feature order:"
    )

    st.write(
        EXPECTED_FEATURE_ORDER
    )

    st.stop()


# ============================================================
# CHECK RAW TRAINING DATA
# ============================================================

missing_raw_columns = [

    variable

    for variable in CONTROL_VARIABLES

    if variable not in raw_X_train.columns

]


if missing_raw_columns:

    st.error(
        "The following controllable variables are missing "
        "from raw_X_train.pkl:"
    )

    for variable in missing_raw_columns:

        st.write(
            f"- {variable}"
        )

    st.stop()


# ============================================================
# HISTORICAL OPERATING BOUNDS
# ============================================================

BOUNDS = []

for variable in CONTROL_VARIABLES:

    lower = float(
        raw_X_train[variable].min()
    )

    upper = float(
        raw_X_train[variable].max()
    )

    BOUNDS.append(
        (lower, upper)
    )


# ============================================================
# HISTORICAL OBSERVED VALUES
#
# Used later for snapping DE solutions to actual
# historically observed operating values.
# ============================================================

HISTORICAL_VALUES = {}

for variable in CONTROL_VARIABLES:

    values = (

        pd.to_numeric(
            raw_X_train[variable],
            errors="coerce"
        )

        .dropna()
        .unique()

    )

    values = np.sort(
        values.astype(float)
    )

    HISTORICAL_VALUES[variable] = values


# ============================================================
# HELPER:
# NORMALIZED PROCESS CHANGE
# ============================================================

def normalized_change(
    current_value,
    candidate_value,
    lower,
    upper
):

    range_value = upper - lower

    if range_value <= 0:

        return 0.0

    return (

        abs(
            candidate_value -
            current_value
        )

        /

        range_value

    )


# ============================================================
# BUILD RF FEATURES
#
# IMPORTANT:
#
# The optimizer changes only:
#
#   C-SLAG
#   SILICA
#   AIR
#   O2
#
# Derived variables are recalculated.
#
# Fe/SiO2
# Fe3O4_Cls
# Matte Grade
#
# are held fixed because we do not have validated
# physical equations for how they change with the
# manipulated variables.
# ============================================================

def build_rf_features(
    current_row,
    candidate_values
):

    row = current_row.copy()


    # --------------------------------------------------------
    # Replace only controllable variables
    # --------------------------------------------------------

    for variable in CONTROL_VARIABLES:

        row[variable] = float(
            candidate_values[variable]
        )


    # --------------------------------------------------------
    # SAFETY CHECKS
    # --------------------------------------------------------

    if row["Matte Grade"] <= 0:

        raise ValueError(
            "Matte Grade must be greater than zero."
        )


    if row["Fe_in_Cl_Slag"] <= 0:

        raise ValueError(
            "Fe_in_Cl_Slag must be greater than zero."
        )


    if row["S"] <= 0:

        raise ValueError(
            "Sulfur concentration must be greater than zero."
        )


    # ========================================================
    # 1. Cu Input
    # ========================================================

    Cu_Input = (

        (
            row["Cu"] / 100.0
        )

        *

        row["CONC. FEED RATE"]

    ) + (

        0.125

        *

        row[
            "C-SLAG FEED RATE - S Furnace"
        ]

    )


    # ========================================================
    # 2. Matte Flow
    # ========================================================

    Matte_Flow = (

        Cu_Input

        /

        (
            row["Matte Grade"] / 100.0
        )

    )


    # ========================================================
    # 3. Fe in Matte
    # ========================================================

    Fe_in_Matte = (

        100.0

        -

        row["Matte Grade"]

        -

        row["S_in_Matte"]

        -

        0.5

    )


    # ========================================================
    # 4. Fe Input
    # ========================================================

    Fe_Input = (

        (
            row["Fe"] / 100.0
        )

        *

        row["CONC. FEED RATE"]

    ) + (

        0.45

        *

        row[
            "C-SLAG FEED RATE - S Furnace"
        ]

    )


    # ========================================================
    # 5. Fe to Matte
    # ========================================================

    Fe_to_Matte = (

        Fe_in_Matte / 100.0

    ) * Matte_Flow


    # ========================================================
    # 6. Fe to Slag
    # ========================================================

    Fe_to_Slag = (

        Fe_Input
        -
        Fe_to_Matte

    )


    # ========================================================
    # 7. Sulfur Feed
    # ========================================================

    Sulfur_Feed = (

        row["S"] / 100.0

    ) * row[
        "CONC. FEED RATE"
    ]


    # ========================================================
    # 8. O2 / Sulfur
    # ========================================================

    O2_to_Sulfur = (

        (

            row["S-FURNACE AIR"]

            *

            0.21

        )

        +

        (

            row["S-FURNACE OXYGEN"]

            *

            0.95

        )

    ) / Sulfur_Feed


    # ========================================================
    # 9. Slag Flow
    # ========================================================

    Slag_Flow = (

        Fe_to_Slag

        /

        (
            row["Fe_in_Cl_Slag"]
            /
            100.0
        )

    )


    # ========================================================
    # NUMERICAL VALIDITY
    # ========================================================

    calculated_values = {

        "Cu_Input":
            Cu_Input,

        "Fe_Input":
            Fe_Input,

        "Sulfur_Feed":
            Sulfur_Feed,

        "Matte_Flow":
            Matte_Flow,

        "Slag_Flow":
            Slag_Flow,

        "O2_to_Sulfur":
            O2_to_Sulfur

    }


    for name, value in calculated_values.items():

        if not np.isfinite(value):

            raise ValueError(
                f"{name} produced an invalid value."
            )


    # ========================================================
    # FINAL RF FEATURE DICTIONARY
    # ========================================================

    feature_dict = {

        "Cu_Input":
            Cu_Input,

        "Fe_Input":
            Fe_Input,

        "Sulfur_Feed":
            Sulfur_Feed,

        "Matte_Flow":
            Matte_Flow,

        "Slag_Flow":
            Slag_Flow,

        "SILICA FEED RATE ":
            row["SILICA FEED RATE "],

        "O2_to_Sulfur":
            O2_to_Sulfur,

        "Fe/SiO2":
            row["Fe/SiO2"],

        "Fe3O4_Cls":
            row["Fe3O4_Cls"],

        "Matte Grade":
            row["Matte Grade"]

    }


    # ========================================================
    # EXACT FEATURE ORDER
    # ========================================================

    X = np.array(

        [

            feature_dict[feature]

            for feature in FEATURE_ORDER

        ],

        dtype=float

    )


    return X


# ============================================================
# RF PROBABILITY
# ============================================================

def get_class_probability(
    X,
    target_class=TARGET_CLASS
):

    X = np.asarray(
        X,
        dtype=float
    ).reshape(1, -1)


    probabilities = rf.predict_proba(
        X
    )[0]


    classes = list(
        rf.classes_
    )


    if target_class not in classes:

        raise ValueError(
            f"Target class {target_class} "
            f"not found in RF classes: {classes}"
        )


    class_index = classes.index(
        target_class
    )


    return float(
        probabilities[class_index]
    )


# ============================================================
# RF PREDICTION
# ============================================================

def get_prediction(X):

    X = np.asarray(
        X,
        dtype=float
    ).reshape(1, -1)


    prediction = rf.predict(
        X
    )[0]


    probabilities = rf.predict_proba(
        X
    )[0]


    probability_dict = {

        int(cls):
            float(prob)

        for cls, prob
        in zip(
            rf.classes_,
            probabilities
        )

    }


    return (
        int(prediction),
        probability_dict
    )


# ============================================================
# COUNTERFACTUAL OBJECTIVE
#
# Objective:
#
#   1. Reach desired probability.
#   2. Minimize process movement.
#
# If target probability is not reached,
# a very large penalty is applied.
# ============================================================

def counterfactual_objective(
    x,
    current_row
):

    # --------------------------------------------------------
    # Candidate dictionary
    # --------------------------------------------------------

    candidate = {

        CONTROL_VARIABLES[i]:
            float(x[i])

        for i in range(
            len(CONTROL_VARIABLES)
        )

    }


    # --------------------------------------------------------
    # Build RF features
    # --------------------------------------------------------

    try:

        X_candidate = build_rf_features(
            current_row,
            candidate
        )

    except Exception:

        return 1e15


    # --------------------------------------------------------
    # Numerical validity
    # --------------------------------------------------------

    if not np.all(
        np.isfinite(
            X_candidate
        )
    ):

        return 1e15


    # --------------------------------------------------------
    # RF probability
    # --------------------------------------------------------

    try:

        probability = get_class_probability(
            X_candidate
        )

    except Exception:

        return 1e15


    # --------------------------------------------------------
    # NORMALIZED PROCESS MOVEMENT
    # --------------------------------------------------------

    total_change_squared = 0.0


    for i, variable in enumerate(
        CONTROL_VARIABLES
    ):

        lower, upper = BOUNDS[i]


        change = normalized_change(

            current_row[variable],

            x[i],

            lower,

            upper

        )


        total_change_squared += (
            change ** 2
        )


    total_change = np.sqrt(
        total_change_squared
    )


    # --------------------------------------------------------
    # TARGET SHORTFALL
    # --------------------------------------------------------

    shortfall = max(

        0.0,

        DESIRED_PROB - probability

    )


    # --------------------------------------------------------
    # FINAL OBJECTIVE
    # --------------------------------------------------------

    objective = (

        CHANGE_WEIGHT
        *
        total_change

    ) + (

        TARGET_PENALTY
        *
        shortfall ** 2

    )


    return float(
        objective
    )


# ============================================================
# SNAP TO HISTORICALLY OBSERVED VALUES
# ============================================================

def snap_to_historical_values(
    candidate
):

    snapped = {}


    for variable in CONTROL_VARIABLES:

        values = HISTORICAL_VALUES[
            variable
        ]


        if len(values) == 0:

            snapped[variable] = float(
                candidate[variable]
            )

            continue


        candidate_value = float(
            candidate[variable]
        )


        nearest_index = np.argmin(

            np.abs(
                values -
                candidate_value
            )

        )


        snapped[variable] = float(
            values[nearest_index]
        )


    return snapped


# ============================================================
# CALCULATE TOTAL NORMALIZED CHANGE
# ============================================================

def calculate_total_change(
    current_row,
    candidate
):

    total = 0.0


    for i, variable in enumerate(
        CONTROL_VARIABLES
    ):

        lower, upper = BOUNDS[i]


        change = normalized_change(

            current_row[variable],

            candidate[variable],

            lower,

            upper

        )


        total += change ** 2


    return float(
        np.sqrt(total)
    )


# ============================================================
# RUN ONE DE OPTIMIZATION
# ============================================================

def run_single_de(
    current_row,
    seed
):

    result = differential_evolution(

        func=lambda x:

            counterfactual_objective(
                x,
                current_row
            ),

        bounds=BOUNDS,

        strategy="best1bin",

        maxiter=MAXITER,

        popsize=POPSIZE,

        tol=1e-7,

        mutation=(0.5, 1.0),

        recombination=0.7,

        seed=seed,

        polish=True,

        updating="immediate",

        workers=1

    )


    candidate = {

        CONTROL_VARIABLES[i]:
            float(result.x[i])

        for i in range(
            len(CONTROL_VARIABLES)
        )

    }


    return candidate


# ============================================================
# COMPLETE COUNTERFACTUAL SEARCH
# ============================================================

def run_counterfactual(
    current_row
):

    # ========================================================
    # CURRENT CONDITION
    # ========================================================

    current_candidate = {

        variable:
            float(current_row[variable])

        for variable
        in CONTROL_VARIABLES

    }


    current_X = build_rf_features(

        current_row,

        current_candidate

    )


    current_prediction, current_probabilities = (
        get_prediction(
            current_X
        )
    )


    current_probability = (
        current_probabilities[
            TARGET_CLASS
        ]
    )


    # ========================================================
    # ALREADY GOOD ENOUGH?
    # ========================================================

    if current_probability >= DESIRED_PROB:

        return {

            "status":
                "TARGET_ALREADY_REACHED",

            "current_prediction":
                current_prediction,

            "current_probability":
                current_probability,

            "counterfactual_probability":
                current_probability,

            "current_candidate":
                current_candidate,

            "counterfactual_candidate":
                current_candidate,

            "changes": {},

            "seed":
                None,

            "all_results": []

        }


    # ========================================================
    # RUN MULTIPLE DE SEEDS
    # ========================================================

    all_results = []


    for seed in SEEDS:

        try:

            raw_candidate = run_single_de(

                current_row,

                seed

            )


            # ------------------------------------------------
            # First evaluate raw DE solution
            # ------------------------------------------------

            raw_X = build_rf_features(

                current_row,

                raw_candidate

            )


            raw_probability = (
                get_class_probability(
                    raw_X
                )
            )


            # ------------------------------------------------
            # Snap to actual historical operating values
            # ------------------------------------------------

            snapped_candidate = (
                snap_to_historical_values(
                    raw_candidate
                )
            )


            # ------------------------------------------------
            # Recalculate RF after snapping
            # ------------------------------------------------

            snapped_X = build_rf_features(

                current_row,

                snapped_candidate

            )


            snapped_probability = (
                get_class_probability(
                    snapped_X
                )
            )


            # ------------------------------------------------
            # Process movement
            # ------------------------------------------------

            total_change = (
                calculate_total_change(

                    current_row,

                    snapped_candidate

                )
            )


            all_results.append({

                "seed":
                    seed,

                "raw_candidate":
                    raw_candidate,

                "raw_probability":
                    raw_probability,

                "candidate":
                    snapped_candidate,

                "probability":
                    snapped_probability,

                "change":
                    total_change,

                "success":
                    snapped_probability >= DESIRED_PROB

            })


        except Exception as e:

            all_results.append({

                "seed":
                    seed,

                "raw_candidate":
                    None,

                "raw_probability":
                    None,

                "candidate":
                    None,

                "probability":
                    None,

                "change":
                    np.inf,

                "success":
                    False,

                "error":
                    str(e)

            })


    # ========================================================
    # REMOVE FAILED RUNS
    # ========================================================

    valid_results = [

        result

        for result
        in all_results

        if result["candidate"] is not None

    ]


    if not valid_results:

        return {

            "status":
                "OPTIMIZATION_FAILED",

            "current_prediction":
                current_prediction,

            "current_probability":
                current_probability,

            "counterfactual_probability":
                None,

            "current_candidate":
                current_candidate,

            "counterfactual_candidate":
                None,

            "changes": {},

            "seed":
                None,

            "all_results":
                all_results

        }


    # ========================================================
    # SUCCESSFUL SOLUTIONS
    #
    # If target is reached:
    #
    # choose the solution requiring the
    # smallest normalized process change.
    # ========================================================

    successful = [

        result

        for result
        in valid_results

        if result["success"]

    ]


    if successful:

        best = min(

            successful,

            key=lambda result:
                result["change"]

        )

        status = "TARGET_REACHED"


    # ========================================================
    # TARGET NOT REACHED
    #
    # Choose highest probability.
    # ========================================================

    else:

        best = max(

            valid_results,

            key=lambda result:
                result["probability"]

        )

        status = "TARGET_NOT_REACHED"


    # ========================================================
    # INDIVIDUAL CHANGES
    # ========================================================

    changes = {}


    for variable in CONTROL_VARIABLES:

        old_value = float(
            current_row[variable]
        )

        new_value = float(
            best["candidate"][variable]
        )


        absolute_change = (
            new_value -
            old_value
        )


        percent_change = 0.0


        if abs(old_value) > 1e-12:

            percent_change = (

                absolute_change
                /
                abs(old_value)

            ) * 100.0


        changes[variable] = {

            "current":
                old_value,

            "counterfactual":
                new_value,

            "absolute_change":
                absolute_change,

            "percent_change":
                percent_change

        }


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "status":
            status,

        "current_prediction":
            current_prediction,

        "current_probability":
            current_probability,

        "counterfactual_probability":
            best["probability"],

        "current_candidate":
            current_candidate,

        "counterfactual_candidate":
            best["candidate"],

        "changes":
            changes,

        "seed":
            best["seed"],

        "normalized_change":
            best["change"],

        "all_results":
            all_results

    }


# ============================================================
# USER INTERFACE
# ============================================================

st.title(
    "CL Slag Copper-Loss Counterfactual Analysis"
)

st.markdown(
    """
This application uses the trained Random Forest classifier to
evaluate the current operating condition and identify feasible
changes to controllable S-furnace variables that increase the
probability of the **lower-Cu-loss class**.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Counterfactual Settings"
)


st.sidebar.write(
    f"Target class: **{TARGET_CLASS}**"
)

st.sidebar.write(
    "Class 1 = Lower Cu loss"
)

st.sidebar.write(
    f"Target probability: **{DESIRED_PROB:.0%}**"
)

st.sidebar.write(
    f"DE iterations: **{MAXITER}**"
)

st.sidebar.write(
    f"DE population size: **{POPSIZE}**"
)

st.sidebar.write(
    f"Seeds: **{SEEDS}**"
)


st.sidebar.markdown(
    "---"
)


st.sidebar.subheader(
    "Model Information"
)


st.sidebar.write(
    f"Model: {type(rf).__name__}"
)

st.sidebar.write(
    f"Training rows: {len(X_train):,}"
)

st.sidebar.write(
    f"RF classes: {list(rf.classes_)}"
)


# ============================================================
# INPUT SECTION
# ============================================================

st.header(
    "1. Current Furnace Operating Condition"
)

st.info(
    "Enter the current furnace condition. "
    "Only C-slag feed, silica feed, furnace air and oxygen "
    "will be changed by the counterfactual optimizer."
)


# ============================================================
# FUNCTION FOR SAFE DEFAULT
# ============================================================

def historical_median(column):

    values = pd.to_numeric(

        raw_X_train[column],

        errors="coerce"

    ).dropna()


    if len(values) == 0:

        return 0.0


    return float(
        values.median()
    )


def historical_min(column):

    values = pd.to_numeric(

        raw_X_train[column],

        errors="coerce"

    ).dropna()


    return float(
        values.min()
    )


def historical_max(column):

    values = pd.to_numeric(

        raw_X_train[column],

        errors="coerce"

    ).dropna()


# ============================================================
# BLEND / CHEMISTRY
# ============================================================

st.subheader(
    "Blend / Feed Chemistry"
)


col1, col2, col3 = st.columns(3)


with col1:

    Cu = st.number_input(

        "Cu (%)",

        value=historical_median("Cu"),

        format="%.4f"

    )


with col2:

    Fe = st.number_input(

        "Fe (%)",

        value=historical_median("Fe"),

        format="%.4f"

    )


with col3:

    S = st.number_input(

        "S (%)",

        value=historical_median("S"),

        format="%.4f"

    )


col1, col2, col3 = st.columns(3)


with col1:

    Al2O3 = st.number_input(

        "Al2O3 (%)",

        value=historical_median("Al2O3"),

        format="%.4f"

    )


with col2:

    CaO = st.number_input(

        "CaO (%)",

        value=historical_median("CaO"),

        format="%.4f"

    )


with col3:

    MgO = st.number_input(

        "MgO (%)",

        value=historical_median("MgO"),

        format="%.4f"

    )


# ============================================================
# CONCENTRATE FEED
# ============================================================

st.subheader(
    "Concentrate Feed"
)


CONC_FEED = st.number_input(

    "Concentrate Feed Rate",

    value=historical_median(
        "CONC. FEED RATE"
    ),

    format="%.4f"

)


# ============================================================
# FIXED PROCESS / ANALYSIS VARIABLES
# ============================================================

st.subheader(
    "Fixed Process / Analysis Variables"
)

st.caption(
    "These variables are held fixed during counterfactual "
    "optimization because the current application does not "
    "contain validated equations describing how they change "
    "with the manipulated furnace variables."
)


col1, col2, col3 = st.columns(3)


with col1:

    S_in_Matte = st.number_input(

        "S in Matte (%)",

        value=historical_median(
            "S_in_Matte"
        ),

        format="%.4f"

    )


with col2:

    Fe_in_Cl_Slag = st.number_input(

        "Fe in CL Slag (%)",

        value=historical_median(
            "Fe_in_Cl_Slag"
        ),

        format="%.4f"

    )


with col3:

    Fe_SiO2 = st.number_input(

        "Fe/SiO2",

        value=historical_median(
            "Fe/SiO2"
        ),

        format="%.4f"

    )


col1, col2 = st.columns(2)


with col1:

    Fe3O4_Cls = st.number_input(

        "Fe3O4 in CL Slag",

        value=historical_median(
            "Fe3O4_Cls"
        ),

        format="%.4f"

    )


with col2:

    Matte_Grade = st.number_input(

        "Matte Grade (%)",

        value=historical_median(
            "Matte Grade"
        ),

        format="%.4f"

    )


# ============================================================
# CONTROLLABLE VARIABLES
# ============================================================

st.header(
    "2. Controllable S-Furnace Variables"
)

st.success(
    """
The counterfactual optimizer is allowed to change only these
four variables. Their values are constrained to the historical
training-data range.
"""
)


# ------------------------------------------------------------
# Current controllable values
# ------------------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    current_cslag = st.number_input(

        "C-Slag Feed Rate - S Furnace",

        min_value=BOUNDS[0][0],

        max_value=BOUNDS[0][1],

        value=float(
            historical_median(
                "C-SLAG FEED RATE - S Furnace"
            )
        ),

        format="%.4f"

    )


with col2:

    current_silica = st.number_input(

        "Silica Feed Rate",

        min_value=BOUNDS[1][0],

        max_value=BOUNDS[1][1],

        value=float(
            historical_median(
                "SILICA FEED RATE "
            )
        ),

        format="%.4f"

    )


col1, col2 = st.columns(2)


with col1:

    current_air = st.number_input(

        "S-Furnace Air",

        min_value=BOUNDS[2][0],

        max_value=BOUNDS[2][1],

        value=float(
            historical_median(
                "S-FURNACE AIR"
            )
        ),

        format="%.4f"

    )


with col2:

    current_oxygen = st.number_input(

        "S-Furnace Oxygen",

        min_value=BOUNDS[3][0],

        max_value=BOUNDS[3][1],

        value=float(
            historical_median(
                "S-FURNACE OXYGEN"
            )
        ),

        format="%.4f"

    )


# ============================================================
# SHOW HISTORICAL RANGES
# ============================================================

with st.expander(
    "View historical operating ranges"
):

    range_data = []

    for variable in CONTROL_VARIABLES:

        range_data.append({

            "Variable":
                variable,

            "Historical Min":
                raw_X_train[variable].min(),

            "Historical Max":
                raw_X_train[variable].max(),

            "Historical Median":
                raw_X_train[variable].median()

        })


    st.dataframe(

        pd.DataFrame(
            range_data
        ),

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# BUILD CURRENT ROW
# ============================================================

current_row = pd.Series({

    "Cu":
        Cu,

    "Fe":
        Fe,

    "S":
        S,

    "Al2O3":
        Al2O3,

    "CaO":
        CaO,

    "MgO":
        MgO,

    "CONC. FEED RATE":
        CONC_FEED,

    "C-SLAG FEED RATE - S Furnace":
        current_cslag,

    "SILICA FEED RATE ":
        current_silica,

    "S-FURNACE AIR":
        current_air,

    "S-FURNACE OXYGEN":
        current_oxygen,

    "S_in_Matte":
        S_in_Matte,

    "Fe_in_Cl_Slag":
        Fe_in_Cl_Slag,

    "Fe/SiO2":
        Fe_SiO2,

    "Fe3O4_Cls":
        Fe3O4_Cls,

    "Matte Grade":
        Matte_Grade

})


# ============================================================
# CURRENT RF PREDICTION
# ============================================================

st.header(
    "3. Current RF Prediction"
)


try:

    current_candidate = {

        variable:
            float(current_row[variable])

        for variable
        in CONTROL_VARIABLES

    }


    current_X = build_rf_features(

        current_row,

        current_candidate

    )


    current_prediction, current_probabilities = (
        get_prediction(
            current_X
        )
    )


    current_lower_probability = (
        current_probabilities[
            TARGET_CLASS
        ]
    )


    current_high_probability = (
        current_probabilities.get(
            0,
            np.nan
        )
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(

            "Predicted Class",

            str(current_prediction)

        )


    with col2:

        st.metric(

            "P(Lower Cu Loss)",

            f"{current_lower_probability:.1%}"

        )


    with col3:

        if current_lower_probability >= DESIRED_PROB:

            st.metric(

                "Target Status",

                "Already Reached"

            )

        else:

            st.metric(

                "Target Status",

                "Optimization Needed"

            )


    # --------------------------------------------------------
    # Probability display
    # --------------------------------------------------------

    st.subheader(
        "RF Class Probabilities"
    )


    probability_df = pd.DataFrame({

        "Class": [

            "Class 0 — Higher Cu Loss",

            "Class 1 — Lower Cu Loss"

        ],

        "Probability": [

            current_probabilities.get(
                0,
                np.nan
            ),

            current_probabilities.get(
                1,
                np.nan
            )

        ]

    })


    probability_df["Probability"] = (

        probability_df["Probability"]
        * 100.0

    )


    st.dataframe(

        probability_df,

        use_container_width=True,

        hide_index=True

    )


except Exception as e:

    st.error(
        f"Could not calculate current RF prediction:\n\n{e}"
    )

    st.stop()


# ============================================================
# CURRENT DERIVED FEATURES
# ============================================================

with st.expander(
    "View current RF input features"
):

    current_feature_df = pd.DataFrame({

        "Feature":
            FEATURE_ORDER,

        "Value":
            current_X

    })


    st.dataframe(

        current_feature_df,

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# COUNTERFACTUAL BUTTON
# ============================================================

st.header(
    "4. Counterfactual Optimization"
)


st.markdown(
    f"""
The optimizer searches for a feasible operating condition with
**P(Class 1) ≥ {DESIRED_PROB:.0%}** while minimizing the total
normalized change in the four controllable furnace variables.

If the current condition already satisfies the target,
no process change is recommended.
"""
)


run_button = st.button(

    "🔍 Find Counterfactual Operating Condition",

    type="primary",

    use_container_width=True

)


# ============================================================
# RUN COUNTERFACTUAL
# ============================================================

if run_button:

    with st.spinner(
        "Running Differential Evolution counterfactual optimization..."
    ):

        try:

            result = run_counterfactual(
                current_row
            )

        except Exception as e:

            st.error(
                f"Counterfactual optimization failed:\n\n{e}"
            )

            st.stop()


    st.session_state[
        "counterfactual_result"
    ] = result


# ============================================================
# DISPLAY RESULT
# ============================================================

if (
    "counterfactual_result"
    in st.session_state
):

    result = st.session_state[
        "counterfactual_result"
    ]


    st.header(
        "5. Counterfactual Result"
    )


    # ========================================================
    # CASE 1:
    # TARGET ALREADY REACHED
    # ========================================================

    if result["status"] == "TARGET_ALREADY_REACHED":

        st.success(

            f"""
**No counterfactual change is required.**

The current condition already has a
**{result['current_probability']:.1%} probability of Class 1
(lower Cu loss)**, which is above the target of
**{DESIRED_PROB:.0%}**.
"""

        )


        st.metric(

            "Current P(Lower Cu Loss)",

            f"{result['current_probability']:.1%}"

        )


    # ========================================================
    # CASE 2:
    #TARGET REACHED
    # ========================================================

    elif result["status"] == "TARGET_REACHED":

        st.success(

            f"""
A feasible counterfactual condition was found.

The optimized condition reaches
**{result['counterfactual_probability']:.1%}**
probability of Class 1 while minimizing process movement.
"""

        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(

                "Current Probability",

                f"{result['current_probability']:.1%}"

            )


        with col2:

            st.metric(

                "Counterfactual Probability",

                f"{result['counterfactual_probability']:.1%}"

            )


        with col3:

            improvement = (

                result[
                    "counterfactual_probability"
                ]

                -

                result[
                    "current_probability"
                ]

            )


            st.metric(

                "Probability Improvement",

                f"{improvement:.1%}"

            )


        # ----------------------------------------------------
        # Changes only
        # ----------------------------------------------------

        st.subheader(
            "Recommended Changes"
        )


        change_rows = []


        for variable, values in (
            result["changes"].items()
        ):

            if abs(
                values["absolute_change"]
            ) > 1e-12:

                change_rows.append({

                    "Variable":
                        variable,

                    "Current":
                        values["current"],

                    "Counterfactual":
                        values["counterfactual"],

                    "Change":
                        values["absolute_change"],

                    "Change (%)":
                        values["percent_change"]

                })


        if change_rows:

            change_df = pd.DataFrame(
                change_rows
            )


            st.dataframe(

                change_df,

                use_container_width=True,

                hide_index=True

            )

        else:

            st.info(
                "No process variable change was required."
            )


        # ----------------------------------------------------
        # Final operating condition
        # ----------------------------------------------------

        st.subheader(
            "Final Counterfactual Operating Condition"
        )


        final_rows = []


        for variable in CONTROL_VARIABLES:

            final_rows.append({

                "Variable":
                    variable,

                "Current":
                    result[
                        "current_candidate"
                    ][variable],

                "Counterfactual":
                    result[
                        "counterfactual_candidate"
                    ][variable]

            })


        st.dataframe(

            pd.DataFrame(
                final_rows
            ),

            use_container_width=True,

            hide_index=True

        )


        st.caption(

            f"Best solution came from DE seed "
            f"{result['seed']} and was snapped to "
            f"historically observed operating values."

        )


    # ========================================================
    # CASE 3:
    # TARGET NOT REACHED
    # ========================================================

    elif result["status"] == "TARGET_NOT_REACHED":

        st.warning(

            f"""
The target probability of **{DESIRED_PROB:.0%}** could not be
reached within the allowed historical operating ranges.

The application is therefore showing the **best feasible
counterfactual found**, rather than claiming that the target
was achieved.
"""

        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(

                "Current Probability",

                f"{result['current_probability']:.1%}"

            )


        with col2:

            st.metric(

                "Best Feasible Probability",

                f"{result['counterfactual_probability']:.1%}"

            )


        with col3:

            improvement = (

                result[
                    "counterfactual_probability"
                ]

                -

                result[
                    "current_probability"
                ]

            )


            st.metric(

                "Probability Improvement",

                f"{improvement:.1%}"

            )


        st.subheader(
            "Best Feasible Changes"
        )


        change_rows = []


        for variable, values in (
            result["changes"].items()
        ):

            if abs(
                values["absolute_change"]
            ) > 1e-12:

                change_rows.append({

                    "Variable":
                        variable,

                    "Current":
                        values["current"],

                    "Best Feasible":
                        values["counterfactual"],

                    "Change":
                        values["absolute_change"],

                    "Change (%)":
                        values["percent_change"]

                })


        if change_rows:

            st.dataframe(

                pd.DataFrame(
                    change_rows
                ),

                use_container_width=True,

                hide_index=True

            )

        else:

            st.info(
                "No meaningful process change was found."
            )


    # ========================================================
    # CASE 4:
    # OPTIMIZATION FAILED
    # ========================================================

    elif result["status"] == "OPTIMIZATION_FAILED":

        st.error(

            """
The counterfactual optimization could not produce a valid
candidate solution. Check the input values and the model
artifacts.
"""

        )


    # ========================================================
    # ALL DE RUNS
    # ========================================================

    if result.get("all_results"):

        with st.expander(
            "View all Differential Evolution runs"
        ):

            run_rows = []


            for run in result[
                "all_results"
            ]:

                run_rows.append({

                    "Seed":
                        run["seed"],

                    "Probability":
                        (
                            run["probability"]
                            if run["probability"]
                            is not None
                            else np.nan
                        ),

                    "Target Reached":
                        run["success"],

                    "Normalized Change":
                        run["change"]

                })


            run_df = pd.DataFrame(
                run_rows
            )


            if not run_df.empty:

                run_df[
                    "Probability"
                ] = (

                    run_df[
                        "Probability"
                    ]
                    * 100.0

                )


                st.dataframe(

                    run_df,

                    use_container_width=True,

                    hide_index=True

                )


# ============================================================
# METHODOLOGY
# ============================================================

st.header(
    "6. Counterfactual Methodology"
)


with st.expander(
    "How the counterfactual calculation works"
):

    st.markdown(
        """
### Step 1 — Current condition

The current furnace condition is converted into the exact
10-feature vector used to train the Random Forest.

### Step 2 — Current RF prediction

The RF calculates:

- Class 0 probability = higher Cu-loss class
- Class 1 probability = lower Cu-loss class

### Step 3 — Check the target

If:

`P(Class 1) ≥ 90%`

then no process change is recommended.

### Step 4 — Differential Evolution

If the current probability is below 90%, Differential Evolution
changes only:

- C-slag feed
- Silica feed
- S-furnace air
- S-furnace oxygen

### Step 5 — Derived features

The following are recalculated for every candidate:

- Cu Input
- Fe Input
- Sulfur Feed
- Matte Flow
- Fe to Matte
- Fe to Slag
- Slag Flow
- O2/Sulfur

### Step 6 — RF evaluation

The resulting 10-feature vector is sent to the trained RF.

### Step 7 — Objective

The optimizer attempts to satisfy:

`P(Class 1) ≥ 90%`

while minimizing normalized movement of the four controllable
variables.

### Step 8 — Historical snapping

The optimized values are snapped to the nearest historically
observed operating values.

### Step 9 — Final verification

The RF prediction is recalculated after snapping.

Therefore, the probability displayed by the application is the
probability of the **final snapped counterfactual condition**,
not merely the raw mathematical solution produced by the
optimizer.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "Random Forest counterfactual analysis | "
    "Class 1 = lower Cu-loss class | "
    f"Target probability = {DESIRED_PROB:.0%}"
)
