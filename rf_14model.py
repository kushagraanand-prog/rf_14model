import streamlit as st
import pandas as pd
import numpy as np
import joblib

from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CL Slag Cu-Loss Counterfactual",
    #page_icon="🔥",
    layout="wide"
)

st.title("CL Slag Cu-Loss Prediction & Counterfactual Analysis")

st.markdown(
    """
This application uses ML Model to identify
changes in controllable S-furnace operating parameters that could move
the current operating condition toward the **low-Cu-loss class**.

"""
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_FILE = "rf_final.pkl"
FEATURE_ORDER_FILE = "feature_order.pkl"
X_TRAIN_FILE = "X_train.npy"
Y_TRAIN_FILE = "y_train.npy"
RAW_X_TRAIN_FILE = "raw_X_train.pkl"

# ============================================================
# EXACT FEATURE ORDER
# ============================================================

FEATURE_ORDER = [
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
# COUNTERFACTUAL VARIABLES
# ============================================================

DECISION_VARIABLES = [
    "C-SLAG FEED RATE - S Furnace",
    "SILICA FEED RATE ",
    "S-FURNACE AIR",
    "S-FURNACE OXYGEN"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_FILE)

    try:
        saved_feature_order = joblib.load(FEATURE_ORDER_FILE)
    except Exception:
        saved_feature_order = FEATURE_ORDER

    X_train = np.load(X_TRAIN_FILE)
    y_train = np.load(Y_TRAIN_FILE)

    try:
        raw_X_train = joblib.load(RAW_X_TRAIN_FILE)
    except Exception:
        raw_X_train = None

    return model, saved_feature_order, X_train, y_train, raw_X_train


try:

    rf_model, saved_feature_order, X_train, y_train, raw_X_train = load_model()

except Exception as e:

    st.error(
        f"""
        Could not load the RF model/artifacts.

        Make sure these files are in the same folder as app.py:

        - {MODEL_FILE}
        - {FEATURE_ORDER_FILE}
        - {X_TRAIN_FILE}
        - {Y_TRAIN_FILE}
        - {RAW_X_TRAIN_FILE}

        Error:
        {e}
        """
    )

    st.stop()


# ============================================================
# FEATURE ORDER SAFETY CHECK
# ============================================================

if list(saved_feature_order) != FEATURE_ORDER:

    st.warning(
        """
        The feature order stored in feature_order.pkl does not exactly
        match the expected 10-feature order.

        The app will use the FEATURE_ORDER defined in this script.
        """
    )


# ============================================================
# ENGINEERING CALCULATIONS
# ============================================================

def calculate_engineered_features(raw):
    """
    Calculate the engineered variables using the SAME equations
    used during RF model training.
    """

    Cu = raw["Cu"]
    Fe = raw["Fe"]
    S = raw["S"]

    conc_feed = raw["CONC. FEED RATE"]
    cslag_feed = raw["C-SLAG FEED RATE - S Furnace"]

    air = raw["S-FURNACE AIR"]
    oxygen = raw["S-FURNACE OXYGEN"]

    matte_grade = raw["Matte Grade"]
    s_in_matte = raw["S_in_Matte"]
    fe_in_cl_slag = raw["Fe_in_Cl_Slag"]

    # --------------------------------------------------------
    # Cu input
    # --------------------------------------------------------

    Cu_Input = (
        (Cu / 100.0) * conc_feed
        +
        0.125 * cslag_feed
    )

    # --------------------------------------------------------
    # Matte flow
    # --------------------------------------------------------

    Matte_Flow = (
        Cu_Input
        /
        (matte_grade / 100.0)
    )

    # --------------------------------------------------------
    # Fe in matte
    # --------------------------------------------------------

    Fe_in_Matte = (
        100.0
        -
        matte_grade
        -
        s_in_matte
        -
        0.5
    )

    # --------------------------------------------------------
    # Fe input
    # --------------------------------------------------------

    Fe_Input = (
        (Fe / 100.0) * conc_feed
        +
        0.45 * cslag_feed
    )

    # --------------------------------------------------------
    # Fe to matte
    # --------------------------------------------------------

    Fe_to_Matte = (
        (Fe_in_Matte / 100.0)
        *
        Matte_Flow
    )

    # --------------------------------------------------------
    # Fe to slag
    # --------------------------------------------------------

    Fe_to_Slag = (
        Fe_Input
        -
        Fe_to_Matte
    )

    # --------------------------------------------------------
    # Sulfur feed
    # --------------------------------------------------------

    Sulfur_Feed = (
        (S / 100.0)
        *
        conc_feed
    )

    # --------------------------------------------------------
    # Oxygen-to-sulfur ratio
    # --------------------------------------------------------

    if Sulfur_Feed <= 0:

        O2_to_Sulfur = np.nan

    else:

        O2_to_Sulfur = (
            (
                air * 0.21
            )
            +
            (
                oxygen * 0.95
            )
        ) / Sulfur_Feed

    # --------------------------------------------------------
    # Slag flow
    # --------------------------------------------------------

    if fe_in_cl_slag <= 0:

        Slag_Flow = np.nan

    else:

        Slag_Flow = (
            Fe_to_Slag
            /
            (fe_in_cl_slag / 100.0)
        )

    return {

        "Cu_Input": Cu_Input,

        "Fe_Input": Fe_Input,

        "Sulfur_Feed": Sulfur_Feed,

        "Matte_Flow": Matte_Flow,

        "Fe_in_Matte": Fe_in_Matte,

        "Fe_to_Matte": Fe_to_Matte,

        "Fe_to_Slag": Fe_to_Slag,

        "O2_to_Sulfur": O2_to_Sulfur,

        "Slag_Flow": Slag_Flow
    }


# ============================================================
# BUILD RF INPUT
# ============================================================

def build_rf_input(raw):

    engineered = calculate_engineered_features(raw)

    features = {

        "Cu_Input":
            engineered["Cu_Input"],

        "Fe_Input":
            engineered["Fe_Input"],

        "Sulfur_Feed":
            engineered["Sulfur_Feed"],

        "Matte_Flow":
            engineered["Matte_Flow"],

        "Slag_Flow":
            engineered["Slag_Flow"],

        "SILICA FEED RATE ":
            raw["SILICA FEED RATE "],

        "O2_to_Sulfur":
            engineered["O2_to_Sulfur"],

        "Fe/SiO2":
            raw["Fe/SiO2"],

        "Fe3O4_Cls":
            raw["Fe3O4_Cls"],

        "Matte Grade":
            raw["Matte Grade"]
    }

    X = pd.DataFrame(
        [[features[f] for f in FEATURE_ORDER]],
        columns=FEATURE_ORDER
    )

    return X, engineered


# ============================================================
# RF PREDICTION
# ============================================================

def predict_rf(raw):

    X, engineered = build_rf_input(raw)

    if X.isna().any().any():

        return None, None, X, engineered

    probabilities = rf_model.predict_proba(X)[0]

    classes = list(rf_model.classes_)

    probability_dict = {
        int(cls): float(prob)
        for cls, prob in zip(classes, probabilities)
    }

    predicted_class = int(
        rf_model.predict(X)[0]
    )

    return (
        predicted_class,
        probability_dict,
        X,
        engineered
    )


# ============================================================
# CURRENT PLANT INPUTS
# ============================================================

st.header("1. Current Plant Operating Condition")

st.markdown(
    "Enter the current measured plant condition."
)


# ------------------------------------------------------------
# Blend composition
# ------------------------------------------------------------

st.subheader("Blend / Feed Composition")

col1, col2, col3, col4 = st.columns(4)

with col1:

    Cu = st.number_input(
        "Cu in Concentrate (%)",
        value=25.0,
        step=0.1,
        format="%.3f"
    )

with col2:

    Fe = st.number_input(
        "Fe in Concentrate (%)",
        value=30.0,
        step=0.1,
        format="%.3f"
    )

with col3:

    S = st.number_input(
        "S in Concentrate (%)",
        value=30.0,
        step=0.1,
        format="%.3f"
    )

with col4:

    conc_feed = st.number_input(
        "Concentrate Feed Rate",
        value=100.0,
        step=1.0,
        format="%.3f"
    )


col1, col2, col3, col4 = st.columns(4)

with col1:

    Al2O3 = st.number_input(
        "Al₂O₃ (%)",
        value=1.0,
        step=0.1,
        format="%.3f"
    )

with col2:

    CaO = st.number_input(
        "CaO (%)",
        value=1.0,
        step=0.1,
        format="%.3f"
    )

with col3:

    MgO = st.number_input(
        "MgO (%)",
        value=1.0,
        step=0.1,
        format="%.3f"
    )

with col4:

    s_in_matte = st.number_input(
        "S in Matte (%)",
        value=0.5,
        step=0.1,
        format="%.3f"
    )


# ------------------------------------------------------------
# Current operating variables
# ------------------------------------------------------------

st.subheader("Current S-Furnace Operating Parameters")

col1, col2 = st.columns(2)

with col1:

    cslag_feed = st.number_input(
        "C-Slag Feed Rate - S Furnace",
        value=20.0,
        step=0.1,
        format="%.3f"
    )

    silica_feed = st.number_input(
        "Silica Feed Rate",
        value=10.0,
        step=0.1,
        format="%.3f"
    )

with col2:

    air = st.number_input(
        "S-Furnace Air Flow",
        value=25000.0,
        step=100.0,
        format="%.3f"
    )

    oxygen = st.number_input(
        "S-Furnace Oxygen Flow",
        value=3000.0,
        step=50.0,
        format="%.3f"
    )


# ------------------------------------------------------------
# Current process/output measurements
# ------------------------------------------------------------

st.subheader("Current Process Measurements")

col1, col2, col3, col4 = st.columns(4)

with col1:

    matte_grade = st.number_input(
        "Matte Grade (%)",
        value=60.0,
        step=0.1,
        format="%.3f"
    )

with col2:

    fe_sio2 = st.number_input(
        "Fe/SiO₂",
        value=1.5,
        step=0.01,
        format="%.4f"
    )

with col3:

    fe3o4_cls = st.number_input(
        "Fe₃O₄ in CL Slag (%)",
        value=20.0,
        step=0.1,
        format="%.3f"
    )

with col4:

    fe_in_cl_slag = st.number_input(
        "Fe in CL Slag (%)",
        value=40.0,
        step=0.1,
        format="%.3f"
    )


# ============================================================
# BUILD CURRENT RAW INPUT
# ============================================================

current_raw = {

    "Cu": Cu,

    "Fe": Fe,

    "S": S,

    "Al2O3": Al2O3,

    "CaO": CaO,

    "MgO": MgO,

    "CONC. FEED RATE": conc_feed,

    "SILICA FEED RATE ": silica_feed,

    "C-SLAG FEED RATE - S Furnace": cslag_feed,

    "S-FURNACE AIR": air,

    "S-FURNACE OXYGEN": oxygen,

    "S_in_Matte": s_in_matte,

    "Fe_in_Cl_Slag": fe_in_cl_slag,

    "Fe/SiO2": fe_sio2,

    "Fe3O4_Cls": fe3o4_cls,

    "Matte Grade": matte_grade
}


# ============================================================
# CURRENT PREDICTION
# ============================================================

current_class, current_probs, current_X, current_engineered = predict_rf(
    current_raw
)


# ============================================================
# DISPLAY CURRENT PREDICTION
# ============================================================

st.header("2. Current Cu-Loss Prediction")

if current_probs is None:

    st.error(
        "The current inputs produced invalid engineered features. "
        "Please check feed rates, Matte Grade and Fe in CL slag."
    )

    st.stop()


p_high = current_probs.get(0, 0.0)
p_low = current_probs.get(1, 0.0)


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Low Cu-Loss Probability",
        f"{p_low * 100:.1f}%"
    )

with col2:

    st.metric(
        "High Cu-Loss Probability",
        f"{p_high * 100:.1f}%"
    )

with col3:

    if current_class == 1:

        st.metric(
            "Current Class",
            "LOW Cu LOSS"
        )

    else:

        st.metric(
            "Current Class",
            "HIGH Cu LOSS"
        )


st.caption(
    "Class 1 = Cu in CL slag 0.70–0.75%. "
    "Class 0 = Cu in CL slag 0.78–0.80%."
)


# ============================================================
# CURRENT ENGINEERING CALCULATIONS
# ============================================================

st.subheader("Current Engineering Calculations")

current_engineering_display = pd.DataFrame({

    "Quantity": [

        "Cu Input",

        "Fe Input",

        "Sulfur Feed",

        "O₂ / Sulfur",

        "Matte Flow",

        "Fe to Matte",

        "Fe to Slag",

        "Slag Flow"
    ],

    "Current Value": [

        current_engineered["Cu_Input"],

        current_engineered["Fe_Input"],

        current_engineered["Sulfur_Feed"],

        current_engineered["O2_to_Sulfur"],

        current_engineered["Matte_Flow"],

        current_engineered["Fe_to_Matte"],

        current_engineered["Fe_to_Slag"],

        current_engineered["Slag_Flow"]
    ]
})

st.dataframe(
    current_engineering_display,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# COUNTERFACTUAL SETTINGS
# ============================================================

st.header("3. Counterfactual Optimization")

st.markdown(
    """
The optimizer changes **only the four controllable operating parameters**.
All other current plant inputs are kept fixed.
"""
)


# ============================================================
# GA SETTINGS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    desired_probability = st.slider(
        "Desired Low-Cu-Loss Probability",
        min_value=0.50,
        max_value=0.99,
        value=0.90,
        step=0.01
    )

with col2:

    maxiter = st.number_input(
        "GA Iterations",
        min_value=20,
        max_value=500,
        value=120,
        step=10
    )

with col3:

    popsize = st.number_input(
        "GA Population Size",
        min_value=5,
        max_value=100,
        value=25,
        step=5
    )


change_penalty_weight = st.slider(
    "Operating-change penalty",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)


# ============================================================
# DETERMINE BOUNDS
# ============================================================

def get_training_bounds():

    bounds = []

    # --------------------------------------------------------
    # If raw training data exists, use actual raw ranges.
    # --------------------------------------------------------

    if isinstance(raw_X_train, pd.DataFrame):

        for variable in DECISION_VARIABLES:

            if variable not in raw_X_train.columns:

                raise ValueError(
                    f"{variable} not found in raw_X_train.pkl"
                )

            low = float(
                raw_X_train[variable].min()
            )

            high = float(
                raw_X_train[variable].max()
            )

            bounds.append(
                (low, high)
            )

    else:

        # ----------------------------------------------------
        # Fallback:
        # use current value ± 20%.
        #
        # Prefer raw_X_train.pkl for real deployment.
        # ----------------------------------------------------

        current_values = [

            current_raw[
                "C-SLAG FEED RATE - S Furnace"
            ],

            current_raw[
                "SILICA FEED RATE "
            ],

            current_raw[
                "S-FURNACE AIR"
            ],

            current_raw[
                "S-FURNACE OXYGEN"
            ]
        ]

        for value in current_values:

            if value == 0:

                bounds.append(
                    (0.0, 1.0)
                )

            else:

                bounds.append(
                    (
                        0.8 * value,
                        1.2 * value
                    )
                )

    return bounds


try:

    bounds = get_training_bounds()

except Exception as e:

    st.error(str(e))
    st.stop()


# ============================================================
# DISPLAY SEARCH BOUNDS
# ============================================================

with st.expander("Show counterfactual search bounds"):

    bounds_df = pd.DataFrame({

        "Parameter": DECISION_VARIABLES,

        "Minimum": [
            b[0] for b in bounds
        ],

        "Maximum": [
            b[1] for b in bounds
        ]
    })

    st.dataframe(
        bounds_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# NORMALIZED OPERATING CHANGE
# ============================================================

def normalized_change(candidate):

    current_values = np.array([

        current_raw[
            "C-SLAG FEED RATE - S Furnace"
        ],

        current_raw[
            "SILICA FEED RATE "
        ],

        current_raw[
            "S-FURNACE AIR"
        ],

        current_raw[
            "S-FURNACE OXYGEN"
        ]

    ], dtype=float)

    candidate = np.array(
        candidate,
        dtype=float
    )

    ranges = np.array([

        bounds[0][1] - bounds[0][0],

        bounds[1][1] - bounds[1][0],

        bounds[2][1] - bounds[2][0],

        bounds[3][1] - bounds[3][0]

    ], dtype=float)

    ranges = np.where(
        ranges <= 0,
        1.0,
        ranges
    )

    normalized = (
        candidate - current_values
    ) / ranges

    return float(
        np.sqrt(
            np.sum(
                normalized ** 2
            )
        )
    )


# ============================================================
# EVALUATE COUNTERFACTUAL CANDIDATE
# ============================================================

def evaluate_candidate(candidate):

    candidate_raw = current_raw.copy()

    candidate_raw[
        "C-SLAG FEED RATE - S Furnace"
    ] = candidate[0]

    candidate_raw[
        "SILICA FEED RATE "
    ] = candidate[1]

    candidate_raw[
        "S-FURNACE AIR"
    ] = candidate[2]

    candidate_raw[
        "S-FURNACE OXYGEN"
    ] = candidate[3]

    try:

        predicted_class, probs, X, engineered = predict_rf(
            candidate_raw
        )

        if probs is None:

            return None

        p_low = probs.get(
            1,
            0.0
        )

        p_high = probs.get(
            0,
            0.0
        )

        distance = normalized_change(
            candidate
        )

        # ----------------------------------------------------
        # Main objective:
        #
        # maximize low-Cu probability
        # while discouraging unnecessary movement.
        # ----------------------------------------------------

        probability_penalty = (
            1.0 - p_low
        )

        objective = (
            probability_penalty
            +
            change_penalty_weight * distance
        )

        return {

            "objective": objective,

            "p_low": p_low,

            "p_high": p_high,

            "predicted_class": predicted_class,

            "distance": distance,

            "raw": candidate_raw,

            "X": X,

            "engineered": engineered
        }

    except Exception:

        return None


# ============================================================
# RUN COUNTERFACTUAL
# ============================================================

run_counterfactual = st.button(
    "🚀 Find Counterfactual Operating Condition",
    type="primary",
    use_container_width=True
)


if run_counterfactual:

    progress = st.progress(
        0
    )

    status = st.empty()

    status.info(
        "Running differential-evolution counterfactual search..."
    )

    # --------------------------------------------------------
    # Objective function
    # --------------------------------------------------------

    def objective_function(candidate):

        result = evaluate_candidate(
            candidate
        )

        if result is None:

            return 1e6

        return result["objective"]


    # --------------------------------------------------------
    # Run GA
    # --------------------------------------------------------

    result = differential_evolution(

        objective_function,

        bounds=bounds,

        strategy="best1bin",

        maxiter=int(maxiter),

        popsize=int(popsize),

        tol=1e-7,

        mutation=(0.5, 1.0),

        recombination=0.7,

        polish=True,

        seed=42,

        updating="immediate",

        workers=1
    )

    progress.progress(
        100
    )

    status.success(
        "Counterfactual optimization completed."
    )


    # ========================================================
    # BEST COUNTERFACTUAL
    # ========================================================

    best_candidate = result.x

    cf_result = evaluate_candidate(
        best_candidate
    )


    if cf_result is None:

        st.error(
            "The optimizer found an invalid candidate."
        )

        st.stop()


    cf_raw = cf_result["raw"]

    cf_engineered = cf_result["engineered"]

    cf_p_low = cf_result["p_low"]

    cf_p_high = cf_result["p_high"]

    cf_class = cf_result["predicted_class"]


    # ========================================================
    # RECOMMENDED OPERATING CHANGES
    # ========================================================

    st.header(
        "4. Recommended Operating Changes"
    )

    st.info(
        """
        These are **model-based counterfactual recommendations**.
        They indicate operating conditions that the trained RF associates
        with a higher probability of the low-Cu-loss class.
        They are not causal guarantees.
        """
    )


    current_values = [

        current_raw[
            "C-SLAG FEED RATE - S Furnace"
        ],

        current_raw[
            "SILICA FEED RATE "
        ],

        current_raw[
            "S-FURNACE AIR"
        ],

        current_raw[
            "S-FURNACE OXYGEN"
        ]
    ]

    cf_values = [

        cf_raw[
            "C-SLAG FEED RATE - S Furnace"
        ],

        cf_raw[
            "SILICA FEED RATE "
        ],

        cf_raw[
            "S-FURNACE AIR"
        ],

        cf_raw[
            "S-FURNACE OXYGEN"
        ]
    ]


    change_df = pd.DataFrame({

        "Operating Parameter": [

            "C-slag feed rate",

            "Silica feed rate",

            "S-furnace air",

            "S-furnace oxygen"
        ],

        "Current": current_values,

        "Counterfactual": cf_values,

        "Change": [

            cf_values[i] - current_values[i]

            for i in range(4)
        ],

        "% Change": [

            (
                (cf_values[i] - current_values[i])
                /
                current_values[i]
                *
                100.0
            )
            if current_values[i] != 0
            else np.nan

            for i in range(4)
        ]
    })


    st.dataframe(
        change_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # CALCULATED ENGINEERING IMPACT
    # ========================================================

    st.header(
        "5. Calculated Engineering Impact"
    )

    st.caption(
        """
        These quantities are recalculated using the engineering equations
        used to construct the RF features. They are not separately
        ML-predicted furnace outputs.
        """
    )


    engineering_comparison = pd.DataFrame({

        "Quantity": [

            "Cu Input",

            "Fe Input",

            "Sulfur Feed",

            "O₂ / Sulfur",

            "Matte Flow",

            "Fe to Matte",

            "Fe to Slag",

            "Slag Flow"
        ],

        "Current": [

            current_engineered["Cu_Input"],

            current_engineered["Fe_Input"],

            current_engineered["Sulfur_Feed"],

            current_engineered["O2_to_Sulfur"],

            current_engineered["Matte_Flow"],

            current_engineered["Fe_to_Matte"],

            current_engineered["Fe_to_Slag"],

            current_engineered["Slag_Flow"]
        ],

        "Counterfactual": [

            cf_engineered["Cu_Input"],

            cf_engineered["Fe_Input"],

            cf_engineered["Sulfur_Feed"],

            cf_engineered["O2_to_Sulfur"],

            cf_engineered["Matte_Flow"],

            cf_engineered["Fe_to_Matte"],

            cf_engineered["Fe_to_Slag"],

            cf_engineered["Slag_Flow"]
        ]
    })


    engineering_comparison["Change"] = (

        engineering_comparison["Counterfactual"]
        -
        engineering_comparison["Current"]
    )


    st.dataframe(
        engineering_comparison,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # CU LOSS PREDICTION IMPACT
    # ========================================================

    st.header(
        "6. Predicted Cu-Loss Outcome"
    )


    probability_comparison = pd.DataFrame({

        "Prediction": [

            "Low Cu-loss probability",

            "High Cu-loss probability"
        ],

        "Current": [

            p_low * 100,

            p_high * 100
        ],

        "Counterfactual": [

            cf_p_low * 100,

            cf_p_high * 100
        ],

        "Change (percentage points)": [

            (cf_p_low - p_low) * 100,

            (cf_p_high - p_high) * 100
        ]
    })


    st.dataframe(
        probability_comparison,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # Probability metrics
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(

            "Low Cu-Loss Probability",

            f"{cf_p_low * 100:.1f}%",

            delta=f"{(cf_p_low - p_low) * 100:+.1f} pp"
        )


    with col2:

        st.metric(

            "High Cu-Loss Probability",

            f"{cf_p_high * 100:.1f}%",

            delta=f"{(cf_p_high - p_high) * 100:+.1f} pp"
        )


    with col3:

        if cf_class == 1:

            st.metric(
                "Counterfactual Class",
                "LOW Cu LOSS"
            )

        else:

            st.metric(
                "Counterfactual Class",
                "HIGH Cu LOSS"
            )


    # ========================================================
    # RF FEATURES
    # ========================================================

    with st.expander(
        "Show RF features used for the counterfactual"
    ):

        st.dataframe(
            cf_result["X"].T.rename(
                columns={0: "Counterfactual Value"}
            ),
            use_container_width=True
        )


    # ========================================================
    # ENGINEERING VALUES
    # ========================================================

    with st.expander(
        "Show detailed engineering calculations"
    ):

        detailed = pd.DataFrame({

            "Variable": list(
                cf_engineered.keys()
            ),

            "Counterfactual Value": list(
                cf_engineered.values()
            )
        })

        st.dataframe(
            detailed,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.header(
        "7. Counterfactual Interpretation"
    )

    st.markdown(
        f"""
### Model-based interpretation

The counterfactual search changed only the four controllable
S-furnace operating parameters.

The RF model's estimated probability of the **low-Cu-loss class**
changed from:

**{p_low * 100:.1f}% → {cf_p_low * 100:.1f}%**

The corresponding high-Cu-loss probability changed from:

**{p_high * 100:.1f}% → {cf_p_high * 100:.1f}%**

The associated changes in Cu input, Fe input, sulfur feed,
O₂/S ratio, Matte Flow and Slag Flow are calculated using the
engineering equations.

**Important:** this analysis identifies a model-based counterfactual
operating condition. It should not be interpreted as experimental
evidence that changing those parameters will causally produce the
predicted Cu-loss change.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CL Slag Cu-Loss Counterfactual Analysis | "
)
