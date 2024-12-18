from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Optional
import shap
import matplotlib.pyplot as plt
import os
import uuid
import zipfile
import io
import json
import dill

from fastapi.middleware.cors import CORSMiddleware

import matplotlib
matplotlib.use("Agg")

app = FastAPI(title="Probability Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Define the input data model without Credit_Score
class ModelInput(BaseModel):
    ID: str
    Customer_ID: str
    Month: str
    Name: Optional[str] = None
    Age: str
    SSN: str
    Occupation: str
    Annual_Income: str
    Monthly_Inhand_Salary: Optional[float] = None
    Num_Bank_Accounts: int
    Num_Credit_Card: int
    Interest_Rate: int
    Num_of_Loan: str
    Type_of_Loan: Optional[str] = None
    Delay_from_due_date: int
    Num_of_Delayed_Payment: Optional[str] = None
    Changed_Credit_Limit: str
    Num_Credit_Inquiries: Optional[float] = None
    Credit_Mix: str
    Outstanding_Debt: str
    Credit_Utilization_Ratio: float
    Credit_History_Age: Optional[str] = None
    Payment_of_Min_Amount: str
    Total_EMI_per_month: float
    Amount_invested_monthly: Optional[str] = None
    Payment_Behaviour: str
    Monthly_Balance: Optional[str] = None

# Global variables to store loaded resources
model = None
preprocessing_pipeline = None
model_step = None
bin_intervals = {}
bin_classification_map = {}

@app.on_event("startup")
def load_resources():
    global model, preprocessing_pipeline, model_step, bin_intervals, bin_classification_map

    with open("model.pkl", "rb") as f:
        full_model = dill.load(f)

    # Split the pipeline and final model step
    preprocessing_pipeline = full_model[:-1]
    model_step = full_model[-1]

    # Load bin intervals
    with open("bin_intervals.json", "r") as f:
        bin_intervals = json.load(f)

    # Define bin classification map
    bin_classification_map = {
        "Grade 1": "Very Reliable",
        "Grade 2": "Reliable",
        "Grade 3": "Average",
        "Grade 4": "Below Average",
        "Grade 5": "Unreliable",
        "Grade 6": "Very Unreliable"
    }

# Helper function to assign bins
def assign_bin(proba: float):
    for bin_num, (low, high) in bin_intervals.items():
        if low <= proba <= high:
            return bin_num, bin_classification_map.get(bin_num, "Unknown")
    return None, "Unknown"

import tempfile
import uuid

@app.post("/predict")
def predict(input: ModelInput):
    try:
        from zipfile import ZipFile
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input.model_dump()])
        # Preprocess and predict
        preprocessed_data = preprocessing_pipeline.transform(input_df)
        proba = model_step.predict_proba(preprocessed_data)[0][1]
        bin_num, classification = assign_bin(proba)
        decision = "approve" if classification in ['Very Reliable', 'Reliable', 'Average'] else "reject"

        # Generate SHAP values
        feature_names = list(preprocessing_pipeline[-1].get_feature_names_out())
        explainer = shap.Explainer(model_step, feature_names=feature_names)
        shap_values = explainer(preprocessed_data)
        explainer_force = shap.Explanation(values=shap_values.values,
                                           base_values=explainer.expected_value,
                                           data=preprocessed_data)
        base_value = explainer.expected_value
        shap_values_array = shap_values.values
        data_row = preprocessed_data

        # Use a stable temporary directory
        temp_dir = tempfile.gettempdir()

        # Use a unique filename to avoid conflicts
        unique_name = f"shap_plots_{uuid.uuid4()}.zip"
        zip_file_path = os.path.join(temp_dir, unique_name)

        # Generate and save plots
        waterfall_path = os.path.join(temp_dir, f"waterfall_{uuid.uuid4()}.png")
        force_path = os.path.join(temp_dir, f"force_{uuid.uuid4()}.png")
        decision_path = os.path.join(temp_dir, f"decision_{uuid.uuid4()}.png")

        # SHAP Waterfall Plot
        plt.figure(figsize=(16 , 10))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.savefig(waterfall_path, bbox_inches='tight')
        plt.close()

        # SHAP Force Plot
        plt.figure(figsize=(30, 6))
        shap.plots.force(explainer_force[0], show=False,matplotlib=True, figsize=(30, 6))
        plt.savefig(force_path, bbox_inches='tight')
        plt.close()

        # SHAP Decision Plot
        plt.figure(figsize=(25 ,10))
        shap.plots.decision(
            base_value=base_value,
            shap_values=shap_values_array[0],
            features=data_row,
            feature_names=feature_names,
            show=False, auto_size_plot=False
        )
        plt.savefig(decision_path, bbox_inches='tight')
        plt.close()

        # Create ZIP file
        with ZipFile(zip_file_path, 'w') as zipf:
            zipf.write(waterfall_path, "shap_waterfall.png")
            zipf.write(force_path, "shap_force.png")
            zipf.write(decision_path, "shap_decision.png")

        # Cleanup individual plot files if desired
        os.remove(waterfall_path)
        os.remove(force_path)
        os.remove(decision_path)

        return {
            "ID": input.ID,
            "proba": f"{round(proba * 100, 2)} %",
            "bin": bin_num,
            "class": classification,
            "decision": decision,
            "shap_plots_download": f"/download_shap_plots/{unique_name}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download_shap_plots/{zip_name}")
def download_shap_plots(zip_name: str):
    file_path = os.path.join(tempfile.gettempdir(), zip_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/zip", filename="shap_plots.zip")
    else:
        raise HTTPException(status_code=404, detail="File not found")


# Endpoint for batch predictions
@app.post("/batch_predict_with_shap")
async def batch_predict_with_shap(file: UploadFile = File(...)):
    try:
        # Load uploaded CSV file
        contents = await file.read()
        input_df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Ensure required columns exist
        required_columns = list(ModelInput.__annotations__.keys())
        if not set(required_columns).issubset(input_df.columns):
            raise HTTPException(status_code=400, detail="Missing required columns in input CSV")

        # Preprocess and predict probabilities
        preprocessed_data = preprocessing_pipeline.transform(input_df)
        probabilities = model_step.predict_proba(preprocessed_data)[:, 1]

        # Assign bins and classifications
        results = []
        for idx, proba in enumerate(probabilities):
            bin_num, classification = assign_bin(proba)
            results.append({
                "ID": input_df.loc[idx, "ID"],
                "proba": proba,
                "bin": bin_num,
                "classification": classification
            })

        # Create predictions CSV
        output_df = pd.DataFrame(results)
        output_csv_path = f"predictions_{uuid.uuid4()}.csv"
        output_df.to_csv(output_csv_path, index=False)

        # Generate SHAP plots for the first 5 rows
        feature_names = preprocessing_pipeline[-1].get_feature_names_out()
        explainer = shap.Explainer(model_step, feature_names=feature_names)
        shap_values = explainer(preprocessed_data)  # SHAP for first 5 rows

        # Paths to SHAP images
        #shap_waterfall_path = f"shap_waterfall.png"
        shap_bar_path = f"shap_bar.png"
        shap_swarm_path = f"shap_swarm.png"

        # Generate SHAP waterfall plot (1st row)
        #plt.figure(figsize=(12, 8))
        #shap.plots.waterfall(shap_values, show=False)
        #plt.savefig(shap_waterfall_path)
        #plt.close()

        # Generate SHAP bar plot
        plt.figure(figsize=(12, 8))
        shap.plots.bar(shap_values, show=True)
        plt.savefig(shap_bar_path)
        plt.close()

        # Generate SHAP swarm plot
        plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(shap_values, show=False)
        plt.savefig(shap_swarm_path)
        plt.close()

        # Create ZIP file with predictions and SHAP plots
        zip_path = f"batch_results_{uuid.uuid4()}.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(output_csv_path, arcname="predictions.csv")
            #zf.write(shap_waterfall_path, arcname="shap_waterfall.png")
            zf.write(shap_bar_path, arcname="shap_bar.png")
            zf.write(shap_swarm_path, arcname="shap_swarm.png")

        # Cleanup temporary files
        os.remove(output_csv_path)
        #os.remove(shap_waterfall_path)
        os.remove(shap_bar_path)
        os.remove(shap_swarm_path)

        # Return the ZIP file as a response
        return FileResponse(zip_path, media_type="application/zip", filename="batch_results_with_shap.zip")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
