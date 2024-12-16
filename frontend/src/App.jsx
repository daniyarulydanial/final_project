import React, { useState } from "react";
import "./App.css";

function App() {
    const [formData, setFormData] = useState({
        ID: "ID_001",
        Customer_ID: "CUST_12345",
        Month: "January",
        Name: "John Doe",
        Age: "35",
        SSN: "123-45-6789",
        Occupation: "Engineer",
        Annual_Income: "75000",
        Monthly_Inhand_Salary: 6250.0,
        Num_Bank_Accounts: 3,
        Num_Credit_Card: 4,
        Interest_Rate: 12,
        Num_of_Loan: "2",
        Type_of_Loan: "Home Loan, Auto Loan",
        Delay_from_due_date: 10,
        Num_of_Delayed_Payment: "5",
        Changed_Credit_Limit: "1000",
        Num_Credit_Inquiries: 2.0,
        Credit_Mix: "Good",
        Outstanding_Debt: "15000",
        Credit_Utilization_Ratio: 30.5,
        Credit_History_Age: "10 Years and 3 Months",
        Payment_of_Min_Amount: "Yes",
        Total_EMI_per_month: 2000.0,
        Amount_invested_monthly: "500",
        Payment_Behaviour: "High_spent_Large_value_payments",
        Monthly_Balance: "1500",
    });

    const [file, setFile] = useState(null);
    const [response, setResponse] = useState(null);
    const [shapPlotUrl, setShapPlotUrl] = useState(null);
    const [zipFileUrl, setZipFileUrl] = useState(null);

    const API_URL = import.meta.env.VITE_BASE_URL;

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({ ...formData, [name]: value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const response = await fetch(`${API_URL}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData),
            });

            if (response.ok) {
                const result = await response.json();
                console.log("Prediction Response:", result);
                setResponse(result);

                // Save SHAP plot URL
                setShapPlotUrl(`${API_URL}/${result.shap_plot_download}`);
            } else {
                alert("Error: Form submission failed");
            }
        } catch (error) {
            console.error("Error:", error);
            alert("An error occurred while submitting the form");
        }
    };

    const handleFileSubmit = async () => {
        try {
            const formDataToSend = new FormData();
            formDataToSend.append("file", file);

            const response = await fetch(`${API_URL}/batch_predict_with_shap`, {
                method: "POST",
                body: formDataToSend,
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);

                // Save ZIP file URL
                setZipFileUrl(url);
            } else {
                alert("Error: File upload failed");
            }
        } catch (error) {
            console.error("Error:", error);
            alert("An error occurred while uploading the file");
        }
    };

    return (
        <div className="App">
            <h1>React Form for Prediction</h1>

            {/* Form Submission */}
            <form onSubmit={handleSubmit}>
                {Object.keys(formData).map((key) => (
                    <div key={key}>
                        <label htmlFor={key}>{key.replace(/_/g, " ")}</label>
                        <input
                            type="text"
                            id={key}
                            name={key}
                            value={formData[key]}
                            onChange={handleChange}
                        />
                    </div>
                ))}
                <button type="submit">Submit Form</button>
            </form>

            {/* File Upload */}
            <label htmlFor="fileUpload">Upload CSV for Batch Processing</label>
            <input
                type="file"
                id="fileUpload"
                onChange={(e) => setFile(e.target.files[0])}
            />
            <button onClick={handleFileSubmit}>Upload File</button>

            {/* Display Prediction Response */}
            {response && (
                <div className="response-container">
                    <h3>Prediction Results</h3>
                    <p>
                        <strong>ID:</strong> {response.ID}
                    </p>
                    <p>
                        <strong>Probability:</strong> {response.proba}
                    </p>
                    <p>
                        <strong>Bin:</strong> {response.bin}
                    </p>
                    <p>
                        <strong>Class:</strong> {response.class}
                    </p>
                    {shapPlotUrl && (
                        <a
                            href={shapPlotUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            Download SHAP Waterfall Plot
                        </a>
                    )}
                </div>
            )}

            {/* Display ZIP File Link */}
            {zipFileUrl && (
                <div className="response-container">
                    <h3>Batch Results</h3>
                    <a href={zipFileUrl} download="batch_results_with_shap.zip">
                        Download Batch Results
                    </a>
                </div>
            )}
        </div>
    );
}

export default App;
