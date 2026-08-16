import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

model = joblib.load('fraud_xgb_model.pkl')
scaler = joblib.load('fraud_scaler.pkl')

st.title("Credit Card Fraud Detection")

mode = st.radio("How do you want to test?", ["Pick a real example", "Enter values manually"])

if mode == "Pick a real example":
    df = pd.read_csv('creditcard.csv')
    examples = pd.concat([
        df[df['Class'] == 1].sample(10, random_state=1),
        df[df['Class'] == 0].sample(10, random_state=1)
    ]).reset_index(drop=True)

    choice = st.selectbox(
        "Choose a transaction to test:",
        options=examples.index,
        format_func=lambda i: f"Transaction {i+1} (actual: {'Fraud' if examples.loc[i, 'Class'] == 1 else 'Normal'})"
    )
    row = examples.loc[[choice]].drop('Class', axis=1)

else:
    st.write("Enter values for each feature (tip: copy real values from creditcard.csv for a meaningful test):")
    time = st.number_input("Time", value=0.0)
    v_values = [st.number_input(f"V{i}", value=0.0, format="%.4f") for i in range(1, 29)]
    amount = st.number_input("Amount", value=0.0)
    row = pd.DataFrame([[time] + v_values + [amount]],
                        columns=['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'])

st.write("Transaction details:", row)

if st.button("Check Transaction"):
    row_scaled = scaler.transform(row)
    prediction = model.predict(row_scaled)[0]
    probability = model.predict_proba(row_scaled)[0][1]
    actual_amount = row['Amount'].values[0]

    st.write(f"**Transaction Amount:** ₹{actual_amount:.2f}")

    if prediction == 1:
        st.error(f"⚠️ Predicted: FRAUD (confidence: {probability:.2%})")
        st.write(f"If this fraud had been missed, the estimated loss would be ₹{actual_amount:.2f}.")
    else:
        st.success(f"✅ Predicted: NORMAL (fraud probability: {probability:.2%})")

    st.subheader("Why did the model decide this?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row_scaled)
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, shap_values[0], feature_names=row.columns, show=False
    )
    st.pyplot(fig)