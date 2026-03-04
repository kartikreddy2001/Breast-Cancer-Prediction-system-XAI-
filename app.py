from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import shap
import joblib
import pandas as pd

app = Flask(__name__)

# ---------------- LOAD MODEL & SCALER ----------------
model = tf.keras.models.load_model("model/deep_nn_model.keras")
scaler = joblib.load("model/scaler.pkl")

# ---------------- LOAD DATA FOR SHAP BACKGROUND ----------------
data = pd.read_csv("model/tumor.csv")

# Safely drop ID column if exists
data = data.drop(columns=["Sample code number"], errors="ignore")

X = data.drop("Class", axis=1)
X_scaled = scaler.transform(X.values)

# Use small background sample for SHAP
background = X_scaled[:50]

# Create SHAP DeepExplainer
explainer = shap.DeepExplainer(model, background)

# Feature names
feature_names = [
    "Clump Thickness",
    "Uniformity of Cell Size",
    "Uniformity of Cell Shape",
    "Marginal Adhesion",
    "Single Epithelial Cell Size",
    "Bare Nuclei",
    "Bland Chromatin",
    "Normal Nucleoli",
    "Mitoses"
]

# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        # Get input values from form
        values = [float(request.form[f]) for f in feature_names]
        input_data = np.array([values])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Prediction
        pred_prob = float(model.predict(input_scaled)[0][0])
        prediction = "Malignant" if pred_prob >= 0.5 else "Benign"

        # ---------------- SHAP Explanation ----------------
        shap_values = explainer.shap_values(input_scaled)

        # Handle DeepExplainer output properly
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.array(shap_values).reshape(-1)

        # Ensure matching length
        if len(shap_values) != len(feature_names):
            shap_values = shap_values[:len(feature_names)]

        # Create dataframe
        contribution_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_values
        })

        # Sort by absolute impact
        contribution_df["Impact"] = contribution_df["SHAP Value"].abs()
        contribution_df = contribution_df.sort_values(by="Impact", ascending=False)

        # Direction column
        contribution_df["Direction"] = contribution_df["SHAP Value"].apply(
            lambda x: "Increases Risk" if x > 0 else "Reduces Risk"
        )

        # Top risk drivers
        top_positive = contribution_df[contribution_df["SHAP Value"] > 0].head(3)

        # Top protective features
        top_negative = contribution_df[contribution_df["SHAP Value"] < 0].head(3)

        # Risk level classification
        if pred_prob > 0.75:
            risk_level = "HIGH"
        elif pred_prob > 0.4:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        return render_template(
            "index.html",
            prediction=prediction,
            probability=round(pred_prob, 3),
            risk_level=risk_level,
            contributions=contribution_df.to_dict(orient="records"),
            top_positive=top_positive.to_dict(orient="records"),
            top_negative=top_negative.to_dict(orient="records")
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
