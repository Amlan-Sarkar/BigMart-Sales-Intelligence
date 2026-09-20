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
    """Load model, encoders, features, and metrics from pickle files."""
    models_dir = Path(__file__).parent / "models"

    try:
        with open(models_dir / "best_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open(models_dir / "label_encoders.pkl", "rb") as f:
            encoders = pickle.load(f)

        with open(models_dir / "feature_names.pkl", "rb") as f:
            features = pickle.load(f)

        with open(models_dir / "metrics.pkl", "rb") as f:
            metrics = pickle.load(f)

        return model, encoders, features, metrics

    except FileNotFoundError as e:
        st.error(f"❌ Error loading artifacts: {e}")
        st.warning("Ensure these files exist in the `models/` folder:\n"
                   "- best_model.pkl\n- label_encoders.pkl\n"
                   "- feature_names.pkl\n- metrics.pkl")
        st.stop()


model, label_encoders, FEATURES, metrics_dict = load_artifacts()

# Find best model name (exclude Baseline)
best_model_name = max(
    (k for k in metrics_dict if k != "Baseline"),
    key=lambda k: metrics_dict[k]["R2"]
)
best_metrics = metrics_dict[best_model_name]
baseline_metrics = metrics_dict.get("Baseline", {})

# ============================================================================
# HEADER
# ============================================================================
st.title("🏬 BigMart Sales Intelligence Predictor")
st.markdown("**Predict item-outlet sales using the trained machine learning model**")
st.divider()

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

        item_visibility = st.slider(
            "Item Visibility",
            min_value=0.0, max_value=0.35, value=0.07, step=0.01,
            help="% of total display area allocated to the product (0 = back-stocked)"
        )

        has_visibility = st.selectbox(
            "Has Shelf Visibility?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No (back-stocked)",
            help="Whether the item has any shelf visibility"
        )

        item_mrp = st.number_input(
            "Item MRP (₹)",
            min_value=0.0, max_value=300.0, value=150.0, step=1.0,
            help="Maximum Retail Price in Indian Rupees"
        )

        # Auto-derive MRP_Segment from MRP for convenience
        if item_mrp <= 70:
            default_mrp_seg = "Budget"
        elif item_mrp <= 140:
            default_mrp_seg = "MidRange"
        else:
            default_mrp_seg = "Premium"

        mrp_segment = st.selectbox(
            "MRP Segment",
            options=["Budget", "MidRange", "Premium"],
            index=["Budget", "MidRange", "Premium"].index(default_mrp_seg),
            help="Price segment (auto-suggested from MRP)"
        )

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

        # Auto-derive Outlet_Age_Group
        if outlet_age <= 5:
            default_age_grp = "New"
        elif outlet_age <= 10:
            default_age_grp = "Mature"
        else:
            default_age_grp = "Established"

        outlet_age_group = st.selectbox(
            "Outlet Age Group",
            options=["New", "Mature", "Established"],
            index=["New", "Mature", "Established"].index(default_age_grp),
            help="Age segment of the outlet"
        )

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
                st.metric("Model R²", f"{best_metrics['R2']:.1%}",
                          help="Variance explained by the model")
            with m3:
                st.metric("Typical Error (MAE)", f"±₹{best_metrics['MAE']:,.0f}",
                          help="Average absolute error on validation data")

            st.markdown("### 📋 Input Summary")
            st.dataframe(
                pd.DataFrame({"Feature": list(input_data.keys()),
                              "Value": list(input_data.values())}),
                use_container_width=True, hide_index=True
            )

            st.markdown("### 💭 Interpretation")
            low = max(0, prediction - best_metrics["MAE"])
            high = prediction + best_metrics["MAE"]
            st.info(
                f"Predicted sales for this product-outlet combination: **₹{prediction:,.0f}**.  \n"
                f"Given the model's typical error (±₹{best_metrics['MAE']:,.0f}), "
                f"expect sales roughly between **₹{low:,.0f}** and **₹{high:,.0f}**."
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
    st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Best Model", best_model_name, f"R² = {best_metrics['R2']:.4f}")
    with c2:
        st.metric("Variance Explained", f"{best_metrics['R2']:.1%}")
    with c3:
        improvement = best_metrics["R2"] - baseline_metrics.get("R2", 0)
        st.metric("Improvement over Baseline", f"+{improvement:.1%}")

    st.divider()

    st.subheader(f"Feature Importance ({best_model_name})")

    try:
        imp_path = Path(__file__).parent / "models" / "feature_importance.csv"
        importance_df = pd.read_csv(imp_path)

        # Use Feature column as labels
        if "Feature" in importance_df.columns:
            importance_df = importance_df.set_index("Feature")

        importance_df = importance_df.sort_values("Importance", ascending=True)

        st.bar_chart(importance_df["Importance"], height=400)

        st.markdown("**Top contributors:**")
        top3 = importance_df.sort_values("Importance", ascending=False).head(3)
        for i, (feat, row) in enumerate(top3.iterrows(), 1):
            st.write(f"{i}. **{feat}** — {row['Importance']*100:.1f}%")

    except FileNotFoundError:
        st.warning("⚠️ feature_importance.csv not found in models/")

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
    """)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("MAE", f"₹{best_metrics['MAE']:,.2f}")
    with m2:
        st.metric("RMSE", f"₹{best_metrics['RMSE']:,.2f}")
    with m3:
        st.metric("R²", f"{best_metrics['R2']:.4f}")

    st.markdown("""
    #### Features Used

    **Numerical**
    - Item Weight, Item Visibility, Item MRP, Outlet Age, Has Visibility

    **Categorical / Engineered**
    - Item Fat Content, Item Type (Grouped), MRP Segment, Outlet Age Group  
    - Outlet Type, Outlet Size, Outlet Location Type

    #### Key Business Insights
    - **Item MRP** is the strongest driver of sales  
    - **Outlet Type** is the second strongest driver  
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