import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

from scipy.optimize import differential_evolution

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CL Slag Cu-Loss Counterfactual",
    #page_icon="🔥",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("CL Slag Cu-Loss Counterfactual Analysis")

st.markdown(
    """
    ### XGBoost-based engineered-feature counterfactual analysis

    This application evaluates how the engineered process-feature state
    can be changed to increase the predicted probability of the
    **lower Cu-loss class** while keeping the blend composition and
    concentrate feed rate constant.
    """
)


# ============================================================
# MODEL / ARTIFACT PATHS
# ============================================================
# ============================================================
# ARTIFACT DIRECTORY
# ============================================================

ARTIFACT_DIR = "streamlit_artifacts"

MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "xgb_final.pkl"
)

FEATURE_ORDER_PATH = os.path.join(
    ARTIFACT_DIR,
    "feature_order.pkl"
)

X_TRAIN_PATH = os.path.join(
    ARTIFACT_DIR,
    "X_train.npy"
)

Y_TRAIN_PATH = os.path.join(
    ARTIFACT_DIR,
    "y_train.npy"
)

RAW_X_TRAIN_PATH = os.path.join(
    ARTIFACT_DIR,
    "raw_X_train.pkl"
)

# ============================================================
# EXACT MODEL FEATURE ORDER
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


# ============================================================
# COUNTERFACTUAL SETTINGS
# ============================================================

TARGET_CLASS = 1
DESIRED_PROB = 0.90

MAXITER = 120
POPSIZE = 25

SEEDS = [42, 43, 44]

TARGET_PENALTY = 1_000_000.0

# Small regularization against unnecessary movement
CHANGE_WEIGHT = 1.0


# ============================================================
# REQUIRED RAW COLUMNS
# ============================================================

REQUIRED_RAW_COLUMNS = [
    "Cu",
    "Fe",
    "S",
    "Al2O3",
    "CaO",
    "MgO",
    "CONC. FEED RATE",
    "C-SLAG FEED RATE - S Furnace",
    "SILICA FEED RATE ",
    "S-FURNACE AIR",
    "S-FURNACE OXYGEN",
    "S_in_Matte",
    "Fe_in_Cl_Slag",
    "Fe/SiO2",
    "Fe3O4_Cls",
    "Matte Grade"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found:\n{os.path.abspath(MODEL_PATH)}"
        )

    return joblib.load(MODEL_PATH)



# ============================================================
# LOAD FEATURE ORDER
# ============================================================

@st.cache_resource
def load_feature_order():

    if not os.path.exists(FEATURE_ORDER_PATH):
        raise FileNotFoundError(
            f"Feature-order file not found:\n"
            f"{os.path.abspath(FEATURE_ORDER_PATH)}"
        )

    return list(
        joblib.load(FEATURE_ORDER_PATH)
    )



# ============================================================
# LOAD TRAINING DATA
# ============================================================

@st.cache_data
def load_training_data():

    required_files = [
        X_TRAIN_PATH,
        Y_TRAIN_PATH,
        RAW_X_TRAIN_PATH
    ]

    for filepath in required_files:

        if not os.path.exists(filepath):

            raise FileNotFoundError(
                f"Required artifact not found:\n"
                f"{os.path.abspath(filepath)}"
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

    if not isinstance(
        raw_X_train,
        pd.DataFrame
    ):
        raw_X_train = pd.DataFrame(
            raw_X_train
        )

    return (
        X_train,
        y_train,
        raw_X_train
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    model = load_model()

    saved_feature_order = (
        load_feature_order()
    )

    X_train, y_train, raw_X_train = (
        load_training_data()
    )

except Exception as e:

    st.error("Error loading artifacts")

    st.exception(e)

    st.write(
        "Current working directory:"
    )

    st.code(
        os.getcwd()
    )

    st.write(
        "Expected artifact files:"
    )

    for filepath in [
        MODEL_PATH,
        FEATURE_ORDER_PATH,
        X_TRAIN_PATH,
        Y_TRAIN_PATH,
        RAW_X_TRAIN_PATH
    ]:

        st.write(
            f"{os.path.abspath(filepath)} "
            f"→ {os.path.exists(filepath)}"
        )

    st.stop()


# ============================================================
# VERIFY FEATURE ORDER
# ============================================================

if saved_feature_order != EXPECTED_FEATURE_ORDER:

    st.error(
        "Feature-order mismatch detected."
    )

    st.write("Expected feature order:")

    st.code(
        "\n".join(
            f"{i + 1}. {x}"
            for i, x in enumerate(EXPECTED_FEATURE_ORDER)
        )
    )

    st.write("Loaded feature order:")

    st.code(
        "\n".join(
            f"{i + 1}. {x}"
            for i, x in enumerate(saved_feature_order)
        )
    )

    st.stop()


FEATURE_ORDER = EXPECTED_FEATURE_ORDER.copy()


# ============================================================
# VERIFY RAW DATA
# ============================================================

missing_raw_columns = [
    col
    for col in REQUIRED_RAW_COLUMNS
    if col not in raw_X_train.columns
]

if missing_raw_columns:

    st.error(
        "The raw_X_train.pkl file is missing required columns."
    )

    st.write(missing_raw_columns)

    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.sidebar:

    st.header("Model")

    st.success("XGBoost Classifier")

    st.write(
        "Target class:"
    )

    st.write(
        "**Class 1 = Lower Cu-loss class**"
    )

    st.write(
        "Class 0 = Higher Cu-loss class"
    )

    st.write(
        f"Desired probability: **{DESIRED_PROB:.0%}**"
    )

    st.divider()

    st.header("Counterfactual philosophy")

    st.write(
        """
        Blend composition and concentrate feed are kept constant.

        The counterfactual searches for a new engineered-feature state
        associated with a higher probability of Class 1.
        """
    )


# ============================================================
# PROCESS EQUATIONS
# ============================================================

def calculate_engineered_features(
    raw_values,
    c_slag_feed,
    silica_feed,
    furnace_air,
    furnace_oxygen,
    fe_sio2,
    fe3o4_cls,
    matte_grade
):
    """
    Calculate the 10 model features.

    Blend composition and concentrate feed remain fixed.

    C-slag, silica, air, oxygen, Fe/SiO2, Fe3O4 and Matte Grade
    are counterfactual variables.

    Matte Grade affects Matte_Flow and therefore Slag_Flow.
    """

    # --------------------------------------------------------
    # Fixed blend / concentrate values
    # --------------------------------------------------------

    Cu = float(raw_values["Cu"])
    Fe = float(raw_values["Fe"])
    S = float(raw_values["S"])

    conc_feed = float(
        raw_values["CONC. FEED RATE"]
    )

    S_in_Matte = float(
        raw_values["S_in_Matte"]
    )

    # --------------------------------------------------------
    # 1. Cu Input
    # --------------------------------------------------------

    Cu_Input = (
        (Cu / 100.0) * conc_feed
        + 0.125 * c_slag_feed
    )

    # --------------------------------------------------------
    # 2. Fe Input
    # --------------------------------------------------------

    Fe_Input = (
        (Fe / 100.0) * conc_feed
        + 0.45 * c_slag_feed
    )

    # --------------------------------------------------------
    # 3. Sulfur Feed
    #
    # Concentrate feed and blend S are fixed.
    # Therefore this remains fixed.
    # --------------------------------------------------------

    Sulfur_Feed = (
        (S / 100.0) * conc_feed
    )

    # --------------------------------------------------------
    # 4. Matte Flow
    # --------------------------------------------------------

    matte_grade_decimal = matte_grade / 100.0

    if matte_grade_decimal <= 0:
        return None

    Matte_Flow = (
        Cu_Input / matte_grade_decimal
    )

    # --------------------------------------------------------
    # 5. Fe in Matte
    # --------------------------------------------------------

    Fe_in_Matte = (
        100.0
        - matte_grade
        - S_in_Matte
        - 0.5
    )

    # --------------------------------------------------------
    # 6. Fe to Matte
    # --------------------------------------------------------

    Fe_to_Matte = (
        (Fe_in_Matte / 100.0)
        * Matte_Flow
    )

    # --------------------------------------------------------
    # 7. Fe to Slag
    # --------------------------------------------------------

    Fe_to_Slag = (
        Fe_Input
        - Fe_to_Matte
    )

    # --------------------------------------------------------
    # 8. Slag Flow
    # --------------------------------------------------------

    fe_slag_decimal = float(
        raw_values["Fe_in_Cl_Slag"]
    ) / 100.0

    if fe_slag_decimal <= 0:
        return None

    Slag_Flow = (
        Fe_to_Slag
        / fe_slag_decimal
    )

    # --------------------------------------------------------
    # 9. O2 to Sulfur
    # --------------------------------------------------------

    if Sulfur_Feed <= 0:
        return None

    O2_to_Sulfur = (
        (
            furnace_air * 0.21
            + furnace_oxygen * 0.95
        )
        / Sulfur_Feed
    )

    # --------------------------------------------------------
    # FINAL MODEL FEATURE VECTOR
    # --------------------------------------------------------

    feature_dict = {

        "Cu_Input": Cu_Input,

        "Fe_Input": Fe_Input,

        "Sulfur_Feed": Sulfur_Feed,

        "Matte_Flow": Matte_Flow,

        "Slag_Flow": Slag_Flow,

        "SILICA FEED RATE ": silica_feed,

        "O2_to_Sulfur": O2_to_Sulfur,

        "Fe/SiO2": fe_sio2,

        "Fe3O4_Cls": fe3o4_cls,

        "Matte Grade": matte_grade
    }

    return feature_dict


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def feature_dict_to_array(feature_dict):

    values = [
        feature_dict[col]
        for col in FEATURE_ORDER
    ]

    return np.asarray(
        values,
        dtype=float
    ).reshape(1, -1)


# ============================================================
# PREDICTION
# ============================================================

def predict_probability(feature_dict):

    X = feature_dict_to_array(
        feature_dict
    )

    probability = model.predict_proba(X)[0]

    return float(
        probability[TARGET_CLASS]
    )


def predict_class(feature_dict):

    X = feature_dict_to_array(
        feature_dict
    )

    return int(
        model.predict(X)[0]
    )


# ============================================================
# CURRENT INPUT
# ============================================================

st.header("1. Current Process Condition")

st.info(
    """
    Enter the current process condition. These values define the
    baseline engineered-feature state from which the counterfactual
    is calculated.
    """
)


# ============================================================
# DEFAULT CURRENT VALUES
# ============================================================

default_values = {}

for col in REQUIRED_RAW_COLUMNS:

    try:
        default_values[col] = float(
            raw_X_train[col].median()
        )

    except Exception:
        default_values[col] = 0.0


# ============================================================
# BLEND COMPOSITION
# ============================================================

st.subheader("Blend Composition — Fixed")

col1, col2, col3 = st.columns(3)

with col1:

    current_Cu = st.number_input(
        "Cu (%)",
        value=float(default_values["Cu"]),
        format="%.4f"
    )

    current_Fe = st.number_input(
        "Fe (%)",
        value=float(default_values["Fe"]),
        format="%.4f"
    )

with col2:

    current_S = st.number_input(
        "S (%)",
        value=float(default_values["S"]),
        format="%.4f"
    )

    current_Al2O3 = st.number_input(
        "Al2O3 (%)",
        value=float(default_values["Al2O3"]),
        format="%.4f"
    )

with col3:

    current_CaO = st.number_input(
        "CaO (%)",
        value=float(default_values["CaO"]),
        format="%.4f"
    )

    current_MgO = st.number_input(
        "MgO (%)",
        value=float(default_values["MgO"]),
        format="%.4f"
    )


# ============================================================
# FIXED CONCENTRATE FEED
# ============================================================

st.subheader("Concentrate Feed — Fixed")

current_conc_feed = st.number_input(
    "CONC. FEED RATE",
    value=float(default_values["CONC. FEED RATE"]),
    format="%.4f"
)


# ============================================================
# OTHER FIXED PROCESS VARIABLES
# ============================================================

st.subheader("Other Process Inputs")

col1, col2 = st.columns(2)

with col1:

    current_S_in_Matte = st.number_input(
        "S_in_Matte",
        value=float(default_values["S_in_Matte"]),
        format="%.4f"
    )

    current_Fe_in_Cl_Slag = st.number_input(
        "Fe_in_Cl_Slag (%)",
        value=float(default_values["Fe_in_Cl_Slag"]),
        format="%.4f"
    )

with col2:

    st.write(
        "The following three variables are allowed to change "
        "during the engineered-feature counterfactual:"
    )

    st.write(
        "- Fe/SiO₂"
    )

    st.write(
        "- Fe₃O₄ in CL slag"
    )

    st.write(
        "- Matte Grade"
    )


# ============================================================
# CURRENT OPERATING VARIABLES
# ============================================================

st.subheader("Current Furnace Operating Condition")

col1, col2 = st.columns(2)

with col1:

    current_cslag = st.number_input(
        "C-Slag Feed Rate",
        value=float(
            default_values[
                "C-SLAG FEED RATE - S Furnace"
            ]
        ),
        format="%.4f"
    )

    current_silica = st.number_input(
        "Silica Feed Rate",
        value=float(
            default_values[
                "SILICA FEED RATE "
            ]
        ),
        format="%.4f"
    )

with col2:

    current_air = st.number_input(
        "S-Furnace Air",
        value=float(
            default_values[
                "S-FURNACE AIR"
            ]
        ),
        format="%.4f"
    )

    current_oxygen = st.number_input(
        "S-Furnace Oxygen",
        value=float(
            default_values[
                "S-FURNACE OXYGEN"
            ]
        ),
        format="%.4f"
    )


# ============================================================
# CURRENT ENGINEERED VARIABLES
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    current_fe_sio2 = st.number_input(
        "Current Fe/SiO₂",
        value=float(
            default_values["Fe/SiO2"]
        ),
        format="%.4f"
    )

with col2:

    current_fe3o4 = st.number_input(
        "Current Fe₃O₄ (%)",
        value=float(
            default_values["Fe3O4_Cls"]
        ),
        format="%.4f"
    )

with col3:

    current_matte_grade = st.number_input(
        "Current Matte Grade (%)",
        value=float(
            default_values["Matte Grade"]
        ),
        format="%.4f"
    )


# ============================================================
# RAW VALUE DICTIONARY
# ============================================================

current_raw_values = {

    "Cu": current_Cu,

    "Fe": current_Fe,

    "S": current_S,

    "Al2O3": current_Al2O3,

    "CaO": current_CaO,

    "MgO": current_MgO,

    "CONC. FEED RATE": current_conc_feed,

    "S_in_Matte": current_S_in_Matte,

    "Fe_in_Cl_Slag": current_Fe_in_Cl_Slag
}


# ============================================================
# CALCULATE CURRENT FEATURES
# ============================================================

current_features = calculate_engineered_features(

    current_raw_values,

    current_cslag,

    current_silica,

    current_air,

    current_oxygen,

    current_fe_sio2,

    current_fe3o4,

    current_matte_grade
)


if current_features is None:

    st.error(
        "Unable to calculate current engineered features. "
        "Please check the entered process values."
    )

    st.stop()


current_probability = predict_probability(
    current_features
)

current_class = predict_class(
    current_features
)


# ============================================================
# CURRENT MODEL RESULT
# ============================================================

st.header("2. Current XGBoost Prediction")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Predicted Class",
        current_class
    )

with col2:

    st.metric(
        "P(Lower Cu-Loss Class)",
        f"{current_probability:.2%}"
    )

with col3:

    if current_class == TARGET_CLASS:

        st.success(
            "Current condition classified as LOWER Cu-loss"
        )

    else:

        st.warning(
            "Current condition classified as HIGHER Cu-loss"
        )


# ============================================================
# CURRENT ENGINEERED FEATURES
# ============================================================

st.subheader("Current Engineered Features")

current_feature_table = pd.DataFrame({

    "Feature": FEATURE_ORDER,

    "Current Value": [
        current_features[x]
        for x in FEATURE_ORDER
    ]
})

st.dataframe(
    current_feature_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# HISTORICAL BOUNDS
# ============================================================

def get_bounds(column):

    series = pd.to_numeric(
        raw_X_train[column],
        errors="coerce"
    ).dropna()

    if len(series) == 0:

        raise ValueError(
            f"No valid values found for {column}"
        )

    return (
        float(series.min()),
        float(series.max())
    )


# ============================================================
# COUNTERFACTUAL VARIABLES
# ============================================================

COUNTERFACTUAL_VARIABLES = [

    "C-SLAG FEED RATE - S Furnace",

    "SILICA FEED RATE ",

    "S-FURNACE AIR",

    "S-FURNACE OXYGEN",

    "Fe/SiO2",

    "Fe3O4_Cls",

    "Matte Grade"
]


# ============================================================
# DISPLAY COUNTERFACTUAL SEARCH SPACE
# ============================================================

st.header("3. Counterfactual Search Space")

st.info(
    """
    The counterfactual keeps the blend composition and concentrate
    feed rate constant.

    Seven variables are allowed to move:
    C-slag, silica, air, oxygen, Fe/SiO₂, Fe₃O₄ and matte grade.

    The remaining engineered features are calculated from the
    process equations.
    """
)


bounds_table = []

for variable in COUNTERFACTUAL_VARIABLES:

    low, high = get_bounds(variable)

    bounds_table.append({

        "Variable": variable,

        "Historical Minimum": low,

        "Historical Maximum": high

    })

st.dataframe(
    pd.DataFrame(bounds_table),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# OPTIMIZATION FUNCTION
# ============================================================

def build_features_from_vector(
    vector,
    raw_values
):

    cslag = vector[0]

    silica = vector[1]

    air = vector[2]

    oxygen = vector[3]

    fe_sio2 = vector[4]

    fe3o4 = vector[5]

    matte_grade = vector[6]

    return calculate_engineered_features(

        raw_values,

        cslag,

        silica,

        air,

        oxygen,

        fe_sio2,

        fe3o4,

        matte_grade
    )


# ============================================================
# BASELINE NORMALIZATION FOR CHANGE PENALTY
# ============================================================

def normalized_change(
    new_value,
    current_value,
    low,
    high
):

    scale = high - low

    if scale <= 0:

        return 0.0

    return (
        (new_value - current_value)
        / scale
    )


# ============================================================
# OBJECTIVE FUNCTION
# ============================================================

def objective_function(
    vector,
    raw_values,
    variable_bounds
):

    try:

        features = build_features_from_vector(
            vector,
            raw_values
        )

        if features is None:

            return TARGET_PENALTY

        probability = predict_probability(
            features
        )

        # ----------------------------------------------------
        # Probability objective
        # ----------------------------------------------------

        objective = -probability

        # ----------------------------------------------------
        # Mild movement penalty
        #
        # This avoids unnecessarily large changes when two
        # solutions have nearly identical probability.
        # ----------------------------------------------------

        current_values = [

            current_cslag,

            current_silica,

            current_air,

            current_oxygen,

            current_fe_sio2,

            current_fe3o4,

            current_matte_grade
        ]

        change_penalty = 0.0

        for i in range(len(vector)):

            low, high = variable_bounds[i]

            change = normalized_change(
                vector[i],
                current_values[i],
                low,
                high
            )

            change_penalty += change ** 2

        objective += (
            CHANGE_WEIGHT
            * 0.01
            * change_penalty
        )

        # ----------------------------------------------------
        # Desired probability penalty
        # ----------------------------------------------------

        if probability < DESIRED_PROB:

            objective += (
                TARGET_PENALTY
                * (DESIRED_PROB - probability)
            )

        return objective

    except Exception:

        return TARGET_PENALTY


# ============================================================
# SNAP TO HISTORICAL OPERATING VALUES
# ============================================================

def snap_to_historical_values(
    vector
):

    snapped = []

    for i, variable in enumerate(
        COUNTERFACTUAL_VARIABLES
    ):

        series = pd.to_numeric(
            raw_X_train[variable],
            errors="coerce"
        ).dropna().values

        if len(series) == 0:

            snapped.append(
                vector[i]
            )

            continue

        nearest_index = np.argmin(
            np.abs(series - vector[i])
        )

        snapped.append(
            float(series[nearest_index])
        )

    return np.asarray(
        snapped,
        dtype=float
    )


# ============================================================
# RUN COUNTERFACTUAL
# ============================================================

run_counterfactual = st.button(
    "🚀 Run Engineered-Feature Counterfactual",
    type="primary",
    use_container_width=True
)


if run_counterfactual:

    st.header(
        "4. Counterfactual Optimization"
    )

    variable_bounds = []

    for variable in COUNTERFACTUAL_VARIABLES:

        variable_bounds.append(
            get_bounds(variable)
        )

    all_results = []

    progress_text = st.empty()

    progress_text.write(
        "Running differential-evolution searches..."
    )

    progress_bar = st.progress(0)

    for seed_index, seed in enumerate(SEEDS):

        try:

            result = differential_evolution(

                func=objective_function,

                bounds=variable_bounds,

                args=(
                    current_raw_values,
                    variable_bounds
                ),

                strategy="best1bin",

                maxiter=MAXITER,

                popsize=POPSIZE,

                tol=0.01,

                mutation=(0.5, 1.0),

                recombination=0.7,

                seed=seed,

                polish=True,

                updating="immediate",

                workers=1
            )

            raw_solution = np.asarray(
                result.x,
                dtype=float
            )

            # ------------------------------------------------
            # Snap solution to historically observed values
            # ------------------------------------------------

            snapped_solution = (
                snap_to_historical_values(
                    raw_solution
                )
            )

            # ------------------------------------------------
            # Recalculate features AFTER snapping
            # ------------------------------------------------

            cf_features = (
                build_features_from_vector(
                    snapped_solution,
                    current_raw_values
                )
            )

            if cf_features is None:

                continue

            cf_probability = (
                predict_probability(
                    cf_features
                )
            )

            cf_class = (
                predict_class(
                    cf_features
                )
            )

            all_results.append({

                "seed": seed,

                "probability": cf_probability,

                "class": cf_class,

                "solution": snapped_solution,

                "features": cf_features,

                "optimizer_success":
                    bool(result.success),

                "optimizer_message":
                    str(result.message)

            })

        except Exception as e:

            st.warning(
                f"Seed {seed} failed: {e}"
            )

        progress_bar.progress(
            (seed_index + 1)
            / len(SEEDS)
        )


    progress_text.empty()


    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if len(all_results) == 0:

        st.error(
            "No valid counterfactual solution was found."
        )

        st.stop()


    # ========================================================
    # SELECT BEST SOLUTION
    # ========================================================

    all_results = sorted(
        all_results,
        key=lambda x: x["probability"],
        reverse=True
    )

    best_result = all_results[0]

    best_solution = best_result[
        "solution"
    ]

    best_features = best_result[
        "features"
    ]

    best_probability = best_result[
        "probability"
    ]

    best_class = best_result[
        "class"
    ]


    # ========================================================
    # COUNTERFACTUAL SUMMARY
    # ========================================================

    st.header(
        "5. Counterfactual Result"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Current P(Class 1)",
            f"{current_probability:.2%}"
        )

    with col2:

        st.metric(
            "Counterfactual P(Class 1)",
            f"{best_probability:.2%}",
            delta=f"{(best_probability-current_probability):.2%}"
        )

    with col3:

        st.metric(
            "Current Class",
            current_class
        )

    with col4:

        st.metric(
            "Counterfactual Class",
            best_class
        )


    # ========================================================
    # PROBABILITY STATUS
    # ========================================================

    if best_probability >= DESIRED_PROB:

        st.success(
            f"""
            Counterfactual target reached.

            Predicted probability of the lower Cu-loss class:
            **{best_probability:.2%}**
            """
        )

    else:

        st.warning(
            f"""
            The desired probability of {DESIRED_PROB:.0%}
            was not reached within the specified historical
            search space.

            Best probability found:
            **{best_probability:.2%}**
            """
        )


    # ========================================================
    # ENGINEERED FEATURE COUNTERFACTUAL TABLE
    # ========================================================

    st.header(
        "6. Engineered-Feature Changes"
    )

    feature_rows = []

    for feature in FEATURE_ORDER:

        current_value = (
            current_features[feature]
        )

        cf_value = (
            best_features[feature]
        )

        change = (
            cf_value
            - current_value
        )

        if abs(current_value) > 1e-12:

            percent_change = (
                change
                / abs(current_value)
                * 100.0
            )

        else:

            percent_change = np.nan

        feature_rows.append({

            "Feature": feature,

            "Current": current_value,

            "Counterfactual": cf_value,

            "Change": change,

            "Change (%)": percent_change

        })


    feature_comparison = pd.DataFrame(
        feature_rows
    )


    st.dataframe(
        feature_comparison.style.format({

            "Current": "{:.4f}",

            "Counterfactual": "{:.4f}",

            "Change": "{:+.4f}",

            "Change (%)": "{:+.2f}%"

        }),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # HIGHLIGHT CHANGING FEATURES
    # ========================================================

    st.subheader(
        "Features that changed"
    )

    changed_rows = []

    for feature in FEATURE_ORDER:

        current_value = (
            current_features[feature]
        )

        cf_value = (
            best_features[feature]
        )

        change = (
            cf_value
            - current_value
        )

        if abs(change) > 1e-9:

            changed_rows.append({

                "Feature": feature,

                "Current": current_value,

                "Counterfactual": cf_value,

                "Change": change

            })


    if len(changed_rows) > 0:

        changed_df = pd.DataFrame(
            changed_rows
        )

        st.dataframe(
            changed_df.style.format({

                "Current": "{:.4f}",

                "Counterfactual": "{:.4f}",

                "Change": "{:+.4f}"

            }),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No engineered feature changed."
        )


    # ========================================================
    # COUNTERFACTUAL RAW SEARCH VARIABLES
    #
    # IMPORTANT:
    # These are shown as search variables, NOT as validated
    # furnace recommendations.
    # ========================================================

    st.header(
        "7. Counterfactual Search Variables"
    )

    st.warning(
        """
        These values are the variables searched by the optimizer.
        They should currently be interpreted as the operating-variable
        representation associated with the counterfactual search,
        not as independently validated furnace recommendations.

        In particular, Fe/SiO₂, Fe₃O₄ and Matte Grade do not yet have
        validated inverse process equations in this application.
        """
    )


    raw_cf_rows = []

    for i, variable in enumerate(
        COUNTERFACTUAL_VARIABLES
    ):

        current_value = [

            current_cslag,

            current_silica,

            current_air,

            current_oxygen,

            current_fe_sio2,

            current_fe3o4,

            current_matte_grade

        ][i]

        cf_value = best_solution[i]

        raw_cf_rows.append({

            "Counterfactual Variable":
                variable,

            "Current":
                current_value,

            "Counterfactual":
                cf_value,

            "Change":
                cf_value - current_value

        })


    raw_cf_df = pd.DataFrame(
        raw_cf_rows
    )


    st.dataframe(
        raw_cf_df.style.format({

            "Current": "{:.4f}",

            "Counterfactual": "{:.4f}",

            "Change": "{:+.4f}"

        }),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # FIXED VARIABLES
    # ========================================================

    st.header(
        "8. Variables Held Constant"
    )

    fixed_rows = [

        {
            "Variable": "Cu",
            "Current": current_Cu,
            "Counterfactual": current_Cu
        },

        {
            "Variable": "Fe",
            "Current": current_Fe,
            "Counterfactual": current_Fe
        },

        {
            "Variable": "S",
            "Current": current_S,
            "Counterfactual": current_S
        },

        {
            "Variable": "Al2O3",
            "Current": current_Al2O3,
            "Counterfactual": current_Al2O3
        },

        {
            "Variable": "CaO",
            "Current": current_CaO,
            "Counterfactual": current_CaO
        },

        {
            "Variable": "MgO",
            "Current": current_MgO,
            "Counterfactual": current_MgO
        },

        {
            "Variable": "CONC. FEED RATE",
            "Current": current_conc_feed,
            "Counterfactual": current_conc_feed
        },

        {
            "Variable": "S_in_Matte",
            "Current": current_S_in_Matte,
            "Counterfactual": current_S_in_Matte
        },

        {
            "Variable": "Fe_in_Cl_Slag",
            "Current": current_Fe_in_Cl_Slag,
            "Counterfactual": current_Fe_in_Cl_Slag
        }

    ]


    fixed_df = pd.DataFrame(
        fixed_rows
    )


    st.dataframe(
        fixed_df.style.format({

            "Current": "{:.4f}",

            "Counterfactual": "{:.4f}"

        }),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # MULTI-SEED RESULTS
    # ========================================================

    st.header(
        "9. Multi-Seed Optimization Results"
    )

    seed_rows = []

    for result in all_results:

        seed_rows.append({

            "Seed":
                result["seed"],

            "P(Class 1)":
                result["probability"],

            "Predicted Class":
                result["class"],

            "Target Reached":
                result["probability"]
                >= DESIRED_PROB,

            "Optimizer Success":
                result["optimizer_success"]

        })


    seed_df = pd.DataFrame(
        seed_rows
    )


    st.dataframe(
        seed_df.style.format({

            "P(Class 1)": "{:.2%}"

        }),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # MODEL FEATURE VECTOR
    # ========================================================

    with st.expander(
        "Show final XGBoost counterfactual feature vector"
    ):

        final_vector = pd.DataFrame({

            "Feature": FEATURE_ORDER,

            "Value": [

                best_features[x]
                for x in FEATURE_ORDER

            ]

        })

        st.dataframe(
            final_vector.style.format({
                "Value": "{:.6f}"
            }),
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # METHODOLOGY
    # ========================================================

    st.header(
        "10. Counterfactual Methodology"
    )

    st.markdown(
        """
        ### Counterfactual formulation

        The trained XGBoost classifier operates on ten engineered
        process features.

        For the counterfactual analysis:

        **Fixed conditions**

        - Blend composition
        - Concentrate feed rate
        - `Sulfur_Feed`, because it is determined by fixed sulfur
          concentration and fixed concentrate feed

        **Counterfactual variables**

        - C-slag feed rate
        - Silica feed rate
        - S-furnace air
        - S-furnace oxygen
        - Fe/SiO₂
        - Fe₃O₄ in CL slag
        - Matte grade

        The engineering equations are then used to calculate the
        corresponding engineered features.

        The resulting ten-feature vector is supplied to the trained
        XGBoost classifier.

        Differential evolution searches the allowed historical
        feature/operating ranges for a counterfactual state that
        increases the probability of the lower Cu-loss class.
        """
    )


    # ========================================================
    # IMPORTANT INTERPRETATION
    # ========================================================

    st.header(
        "11. Interpretation"
    )

    st.info(
        """
        **Important:** this analysis is currently an engineered-feature
        counterfactual, not a final validated furnace operating
        recommendation.

        Fe/SiO₂, Fe₃O₄ and Matte Grade are allowed to change because
        they are direct inputs to the XGBoost model. However, validated
        process relationships linking these three quantities back to
        specific furnace control actions have not been imposed here.

        Therefore, the result should be interpreted as:

        **"What engineered-feature state is associated with a higher
        predicted probability of lower Cu loss?"**

        rather than:

        **"These exact furnace changes will definitely produce that
        engineered-feature state."**
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
    XGBoost CL Slag Cu-Loss Counterfactual Analysis |
    Engineered-feature optimization with fixed blend composition
    and concentrate feed
    """
)
