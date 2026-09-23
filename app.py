"""
BigMart Sales Intelligence & Prediction - Streamlit App
Works with the improved notebook that uses engineered features
and saves best_model.pkl + dynamic metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(
    page_title="BigMart Sales Predictor",
    page_icon="🏬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { padding-top: 1.5rem; }
    .stMetric { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD ARTIFACTS
# ============================================================================
@st.cache_resource
def load_artifacts():
    """Load model, encoders, features, metrics, and feature bins from pickle files."""
    models_dir = Path(__file__).parent / "models"

    try:
        with open(models_dir / "best_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open(models_dir / "label_encoders.pkl", "rb") as f:
            encoders = pickle.load(f)

        with open(models_dir / "feature_names.pkl", "rb") as f:
            features = pickle.load(f)

        with open(models_dir / "metrics.pkl", "rb") as f:
            metrics_data = pickle.load(f)

        with open(models_dir / "feature_bins.pkl", "rb") as f:
            feature_bins = pickle.load(f)

        # Support the new metrics.pkl structure:
        # {
        #     "holdout": {...},
        #     "grouped_cv": {...}
        # }
        metrics = metrics_data["holdout"]
        grouped_cv_metrics = metrics_data["grouped_cv"]

        return model, encoders, features, metrics, grouped_cv_metrics, feature_bins

    except FileNotFoundError as e:
        st.error(f"❌ Error loading artifacts: {e}")
        st.warning(
            "Ensure these files exist in the `models/` folder:\n"
            "- best_model.pkl\n"
            "- label_encoders.pkl\n"
            "- feature_names.pkl\n"
            "- metrics.pkl\n"
            "- feature_bins.pkl"
        )
        st.stop()
    except (pickle.UnpicklingError, KeyError, EOFError, AttributeError, ModuleNotFoundError) as e:
        # Covers a stale/corrupt pickle, a metrics.pkl missing the expected
        # "holdout"/"grouped_cv" keys, or a scikit-learn/library version
        # mismatch between the environment that trained the model and the
        # one serving it (a common cause of unpicklable model objects).
        st.error(f"❌ Could not load model artifacts: {e}")
        st.warning(
            "This usually means the files in `models/` were produced by a "
            "different version of the notebook/environment than the one "
            "running this app, or a file is corrupted. Re-run the "
            "notebook's deployment cell (Section 13) with matching "
            "package versions and retry."
        )
        st.stop()


@st.cache_data
def load_feature_importance():
    """Load feature importance table, sorted descending by importance."""
    imp_path = Path(__file__).parent / "models" / "feature_importance.csv"
    df = pd.read_csv(imp_path)

    required_columns = {"Feature", "Importance"}

    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(
            f"feature_importance.csv is missing required column(s): "
            f"{', '.join(sorted(missing))}"
        )

    df = df.set_index("Feature")
    return df.sort_values("Importance", ascending=False)


model, label_encoders, FEATURES, metrics_dict, grouped_cv_metrics, feature_bins = load_artifacts()

try:
    importance_df_desc = load_feature_importance()
    top3_features = importance_df_desc.head(3)
except (FileNotFoundError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
    importance_df_desc = None
    top3_features = None

if importance_df_desc is None:
    st.warning(
        "⚠️ Feature importance data is unavailable or has an invalid format."
    )


# Find best model name — by GROUPED-CV R², not holdout R².
#
# The holdout split in the notebook is a plain random 80/20 split, so the
# same product can appear in both halves; a model can partly "memorize"
# a product's typical sales through correlated features, which inflates
# holdout R². Grouped-CV (grouped by product) is the honest estimate of
# how the model will do on a product it has never seen — the scenario
# this app is actually used for — so that's what selects the model shown.
candidate_names = [k for k in grouped_cv_metrics if k != "Baseline"]

if not candidate_names:
    st.error(
        "❌ No candidate models found in the saved Grouped-CV metrics. "
        "Re-run the notebook's Cross-Validation section (Section 9) and "
        "the deployment cell (Section 13)."
    )
    st.stop()

best_model_name = max(candidate_names, key=lambda k: grouped_cv_metrics[k]["R2"])

cv_metrics = grouped_cv_metrics[best_model_name]
holdout_metrics = metrics_dict.get(best_model_name, cv_metrics)
baseline_metrics = metrics_dict.get("Baseline", {})
error_quantiles = metrics_dict.get("error_quantiles")

# ============================================================================
# HEADER
# ============================================================================
st.title("🏬 BigMart Sales Intelligence Predictor")
st.markdown("**Predict item-outlet sales using the trained machine learning model**")
st.divider()

def warn_if_outside_training_range(value, range_key, label):
    """Flag when an input is outside the range the model was actually
    trained on. The model can still produce a number outside this range,
    but it's extrapolating rather than interpolating, so accuracy is less
    trustworthy there. No-op if this feature_bins.pkl predates the fix
    that started saving these ranges."""
    training_range = feature_bins.get(range_key)
    if not training_range:
        return
    lo, hi = training_range
    if value < lo or value > hi:
        st.caption(
            f"⚠️ {label} = {value:g} is outside the training range "
            f"[{lo:g}, {hi:g}] — the model is extrapolating here."
        )


tab1, tab2, tab3 = st.tabs(["💡 Make Prediction", "📊 Model Performance", "ℹ️ About"])

# ============================================================================
# TAB 1: PREDICTION
# ============================================================================
with tab1:
    st.subheader("Enter Product & Outlet Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Product Details**")

        item_weight = st.number_input(
            "Item Weight (kg)",
            min_value=0.0, max_value=50.0, value=12.65, step=0.1,
            help="Weight of the product in kilograms"
        )
        warn_if_outside_training_range(item_weight, "item_weight_range", "Item Weight")

        item_visibility = st.slider(
            "Item Visibility",
            min_value=0.0, max_value=0.35, value=0.07, step=0.01,
            help="% of total display area allocated to the product (0 = back-stocked)"
        )
        warn_if_outside_training_range(item_visibility, "item_visibility_range", "Item Visibility")

        # Automatically derive Has_Visibility from Item Visibility
        has_visibility = int(item_visibility > 0)

        st.caption(
            f"Has Shelf Visibility: **{'Yes' if has_visibility else 'No (back-stocked)'}**"
     )

        item_mrp = st.number_input(
            "Item MRP (₹)",
            min_value=0.0, max_value=270.0, value=150.0, step=1.0,
            help="Maximum Retail Price in Indian Rupees"
        )
        warn_if_outside_training_range(item_mrp, "item_mrp_range", "Item MRP")

        # Auto-derive MRP_Segment using saved feature bin definitions
        mrp_segment = pd.cut(
            [item_mrp],
            bins=feature_bins["mrp_bins"],
            labels=feature_bins["mrp_labels"],
            include_lowest=True
        )[0]

        mrp_segment = str(mrp_segment)

        st.caption(f"MRP Segment: **{mrp_segment}**")

        item_fat_content = st.selectbox(
            "Item Fat Content",
            options=sorted(label_encoders["Item_Fat_Content"].classes_.tolist()),
            help="Dietary classification"
        )

        item_type_grouped = st.selectbox(
            "Item Type (Grouped)",
            options=sorted(label_encoders["Item_Type_Grouped"].classes_.tolist()),
            help="Product category (rare types grouped as 'Other')"
        )

    with col2:
        st.markdown("**Outlet Details**")

        outlet_age = st.slider(
            "Outlet Age (years)",
            min_value=0, max_value=30, value=15, step=1,
            help="Years since establishment (reference year 2013)"
        )
        warn_if_outside_training_range(outlet_age, "outlet_age_range", "Outlet Age")

        # Auto-derive Outlet_Age_Group using saved feature bin definitions
        outlet_age_group = pd.cut(
            [outlet_age],
            bins=feature_bins["age_bins"],
            labels=feature_bins["age_labels"],
            include_lowest=True
        )[0]

        outlet_age_group = str(outlet_age_group)

        st.caption(f"Outlet Age Group: **{outlet_age_group}**")

        outlet_size = st.selectbox(
            "Outlet Size",
            options=sorted(label_encoders["Outlet_Size"].classes_.tolist()),
            help="Physical size of the outlet"
        )

        outlet_type = st.selectbox(
            "Outlet Type",
            options=sorted(label_encoders["Outlet_Type"].classes_.tolist()),
            help="Type of retail outlet"
        )

        outlet_location = st.selectbox(
            "Outlet Location Type",
            options=sorted(label_encoders["Outlet_Location_Type"].classes_.tolist()),
            help="Metropolitan tier of the outlet location"
        )

    st.divider()

    if st.button("🎯 Predict Sales", type="primary", use_container_width=True):
        input_data = {
            "Item_Weight": item_weight,
            "Item_Visibility": item_visibility,
            "Item_MRP": item_mrp,
            "Outlet_Age": outlet_age,
            "Has_Visibility": has_visibility,
            "Item_Fat_Content": item_fat_content,
            "Item_Type_Grouped": item_type_grouped,
            "MRP_Segment": mrp_segment,
            "Outlet_Age_Group": outlet_age_group,
            "Outlet_Type": outlet_type,
            "Outlet_Size": outlet_size,
            "Outlet_Location_Type": outlet_location,
        }

        df_input = pd.DataFrame([input_data])

        # Encode categorical features
        categorical_cols = [
            "Item_Fat_Content", "Item_Type_Grouped", "MRP_Segment",
            "Outlet_Age_Group", "Outlet_Type", "Outlet_Size", "Outlet_Location_Type"
        ]

        for col in categorical_cols:
            try:
                df_input[col] = label_encoders[col].transform(df_input[col].astype(str))
            except ValueError as e:
                st.error(f"❌ Unknown value for {col}: {e}")
                st.stop()

        # Keep only the features the model expects, in the correct order
        df_input = df_input[FEATURES]

        try:
            prediction = model.predict(df_input)[0]

            st.success("✅ Prediction Generated")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Predicted Sales", f"₹{prediction:,.2f}")
            with m2:
                st.metric("Model R² (unseen products)", f"{cv_metrics['R2']:.1%}",
                          help="Grouped 5-fold CV, grouped by product — variance "
                               "explained when the model has never seen this "
                               "product before. This is the more honest number "
                               "for a brand-new product; see the Model "
                               "Performance tab for the (higher) holdout figure.")
            with m3:
                st.metric("Typical Error, unseen products (MAE)", f"±₹{cv_metrics['MAE']:,.0f}",
                          help="Average absolute error, grouped cross-validation.")

            st.markdown("### 📋 Input Summary")
            st.dataframe(
                pd.DataFrame({"Feature": list(input_data.keys()),
                              "Value": list(input_data.values())}),
                use_container_width=True, hide_index=True
            )

            st.markdown("### 💭 Interpretation")
            if error_quantiles is not None:
                # Empirical 80% interval from the actual holdout residual
                # distribution, rather than a symmetric +/-MAE band. Sales
                # are right-skewed, so the true error spread is not
                # symmetric around the prediction — this interval reflects
                # that, at the cost of still being holdout-sample-sized
                # (i.e. approximate, not a formal statistical guarantee).
                low = max(0, prediction + error_quantiles["p10"])
                high = prediction + error_quantiles["p90"]
                interval_note = "an empirical 80% interval from holdout prediction errors"
            else:
                # Fallback for a metrics.pkl saved before this interval fix existed.
                low = max(0, prediction - cv_metrics["MAE"])
                high = prediction + cv_metrics["MAE"]
                interval_note = "a rough ±MAE band (re-run the notebook's deployment cell for a better-calibrated interval)"

            st.info(
                f"Predicted sales for this product-outlet combination: **₹{prediction:,.0f}**.  \n"
                f"Based on {interval_note}, "
                f"expect sales roughly between **₹{low:,.0f}** and **₹{high:,.0f}**. "
                f"This is an approximate range, not a formal confidence interval."
            )

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# ============================================================================
# TAB 2: MODEL PERFORMANCE
# ============================================================================
with tab2:
    st.subheader("Model Performance Summary")

    metrics_rows = []
    for name, m in metrics_dict.items():
        metrics_rows.append({
            "Model": name,
            "MAE (₹)": f"{m['MAE']:,.2f}",
            "RMSE (₹)": f"{m['RMSE']:,.2f}",
            "R² Score": f"{m['R2']:.4f}"
        })

    st.dataframe(
        pd.DataFrame(metrics_rows),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.caption(
        f"**{best_model_name}** was selected by grouped-CV R² "
        "(generalization to unseen products), not by the holdout R² below — "
        "see the Grouped Cross-Validation section for why."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Best Model",
            best_model_name,
            f"Holdout R² = {holdout_metrics['R2']:.4f}"
        )

    with c2:
        st.metric(
            "Variance Explained (holdout)",
            f"{holdout_metrics['R2']:.1%}"
        )

    with c3:
        improvement = holdout_metrics["R2"] - baseline_metrics.get("R2", 0)
        st.metric(
            "Improvement over Baseline (holdout)",
            f"+{improvement:.1%}"
        )

    # ========================================================================
    # CROSS-VALIDATION
    # ========================================================================
    st.divider()

    st.subheader("Cross-Validation")

    st.caption(
        f"5-Fold Grouped Cross-Validation — grouped by Product ({best_model_name}). "
        "This is the metric that decided which model is deployed above, because it "
        "holds out whole products rather than individual rows — a plain random "
        "80/20 split can leak a product's other outlet-records into both sides."
    )

    cv1, cv2, cv3 = st.columns(3)

    with cv1:
        st.metric(
            "R²",
            f"{cv_metrics['R2']:.4f}"
        )

    with cv2:
        st.metric(
            "MAE",
            f"₹{cv_metrics['MAE']:,.0f}"
        )

    with cv3:
        st.metric(
            "RMSE",
            f"₹{cv_metrics['RMSE']:,.0f}"
        )

    st.caption(
        "Each fold holds out complete products, measuring generalization to unseen products. "
        "Note this R² is typically lower than the holdout R² above — that's expected, and is "
        "the more trustworthy number for predicting a brand-new product."
    )

    # ========================================================================
    # FEATURE IMPORTANCE
    # ========================================================================
    st.divider()

    st.subheader(f"Feature Importance ({best_model_name})")

    if importance_df_desc is not None:
        st.bar_chart(
            importance_df_desc["Importance"].sort_values(ascending=True),
            height=400
        )

        st.markdown("**Top contributors:**")

        for i, (feat, row) in enumerate(top3_features.iterrows(), 1):
            st.write(
                f"{i}. **{feat}** — {row['Importance'] * 100:.1f}%"
            )

    else:
        st.warning(
            "⚠️ feature_importance.csv not found in models/"
        )

# ============================================================================
# TAB 3: ABOUT
# ============================================================================
with tab3:
    st.subheader("About This Model")

    st.markdown(f"""
    ### BigMart Sales Intelligence & Prediction

    This app uses a trained **{best_model_name}** model to predict item-outlet sales
    for BigMart retail stores.

    #### Dataset
    - **Source:** Kaggle BigMart Sales Data  
    - **Training samples:** 8,523 transactions  
    - **Products:** 1,559 unique items  
    - **Outlets:** 10 retail stores  
    - **Target:** Item_Outlet_Sales (₹)

    #### Model Details
    - **Algorithm:** {best_model_name}
    - **Features:** {len(FEATURES)} (numerical + categorical + engineered)
    - **Evaluation:** Hold-out validation + 5-Fold GroupKFold (by product)

    #### Current Performance
    *(Grouped 5-fold CV — estimated performance on products the model has never seen)*
    """)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("MAE", f"₹{cv_metrics['MAE']:,.2f}")
    with m2:
        st.metric("RMSE", f"₹{cv_metrics['RMSE']:,.2f}")
    with m3:
        st.metric("R²", f"{cv_metrics['R2']:.4f}")

    st.caption(
        f"Holdout validation (products may repeat across train/test): "
        f"MAE ₹{holdout_metrics['MAE']:,.2f}, R² {holdout_metrics['R2']:.4f} — "
        "see the Model Performance tab for the full comparison."
    )

    st.markdown("""
    #### Features Used

    **Numerical**
    - Item Weight, Item Visibility, Item MRP, Outlet Age, Has Visibility

    **Categorical / Engineered**
    - Item Fat Content, Item Type (Grouped), MRP Segment, Outlet Age Group  
    - Outlet Type, Outlet Size, Outlet Location Type

    #### Key Business Insights
    """)

    if top3_features is not None:
        rank_labels = ["strongest", "second strongest", "third strongest"]
        for (feat, row), label in zip(top3_features.iterrows(), rank_labels):
            st.markdown(f"- **{feat}** is the {label} driver of sales")
    else:
        st.markdown("- Feature importance data unavailable (feature_importance.csv not found)")

    st.markdown("""
    - Model is more reliable for known products; for brand-new products combine  
      the prediction with baseline / expert judgment  

    #### Limitations
    - No brand, promotion, or seasonality features  
    - No temporal (month / holiday) signals  
    - Reference year is 2013 — recalibrate if using much newer data  

    ---
 
    **Author:** Amlan Sarkar  
    **Data:** [Kaggle BigMart Sales](https://www.kaggle.com/brijbhushannanda1979/bigmart-sales-data)
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.caption(f"🔧 Built with Streamlit  |  🤖 Model: {best_model_name}  |  📊 Data: Kaggle BigMart")