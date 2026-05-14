# 🚲 BikeShop Marketing Mix Model Director

An interactive, high-fidelity Marketing Mix Modeling (MMM) application designed for strategic budget optimization. This tool leverages **Google Meridian** for Bayesian inference and **PySiMMMulator** for high-fidelity marketing data simulation.

## 🌟 Key Features

The application is structured into four progressive stages of marketing analysis:

### 🛠️ Stage 1: Marketing Landscape Configuration
*   **Detailed Global Setup**: Configure years of data, seasonality variance, and growth trends.
*   **Advanced Channel Modeling**: Define channel types (Impressions/Clicks), True CVR, and Cost (CPM/CPC).
*   **Native Modeling**: Control **Geometric Adstock (Lambda)** and **Hill Saturation (Alpha/Gamma)** parameters with real-time preview charts.
*   **Bayesian Training**: Execute actual Google Meridian MCMC sampling to learn parameters from historical data.

### 📈 Stage 2: Investment Simulator
*   **"What-If" Analysis**: Adjust total weekly investment and percentage-based channel allocations in real-time.
*   **ROI Alignment**: Visualize where your current spend sits on the learned saturation curves.
*   **Channel Locking**: Fix strategic budget mandates (e.g., "Brand must be 20%") and optimize the remainder.
*   **One-Click Optimization**: Use the 🚀 **Optimize Allocation** button to find the mathematically ideal mix.

### 🔍 Stage 3: Data Explorer
*   **Synthetic Data Audit**: Fetch and visualize the underlying daily/weekly historical data.
*   **Dual-Axis Charting**: Directly compare marketing activity (Spend/Impressions) against total bike purchases over time.

### 🎯 Stage 4: Marginal Efficiency Analysis
*   **Budget Sweep**: Automatically simulate 100 discrete optimization points across a custom budget range.
*   **CPA Efficiency Curve**: Identify the point of diminishing returns where acquisition costs begin to spike.
*   **Strategic Stacked Mix**: Visualize how the optimal budget distribution shifts between channels as you scale your business.

---

## 🏗️ Architecture

The project follows a modular 3-tier architecture:

1.  **Wrapper Layer (`wrapper/`)**: Manages the integration with PySiMMMulator for data generation and Google Meridian for model training and analysis.
2.  **API Layer (`api/`)**: A RESTful service built with Starlette that exposes the model's capabilities to the front-end.
3.  **UI Layer (`ui/`)**: A professional, Google-branded dashboard built with Streamlit and Plotly.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have the required virtual environment with `meridian`, `pysimmmulator`, `streamlit`, and `starlette` installed.

### 2. Launch the Application
Use the unified startup script to launch both the backend and frontend simultaneously:

```bash
./run.sh [optional_path_to_venv]
```

*By default, the script looks for a virtual environment at `../bikevenv`.*

### 3. Access the Dashboard
Once started, open your browser and navigate to:
**`http://localhost:8501`**

---

## 🛠️ Technical Stack

*   **Model**: [Google Meridian](https://github.com/google/meridian) (Bayesian MMM)
*   **Simulation**: [PySiMMMulator](https://github.com/google/pysimmmulator)
*   **API**: Starlette / Uvicorn
*   **UI**: Streamlit
*   **Visualizations**: Plotly / Graph Objects
*   **Optimization**: SciPy (SLSQP)

---

## 🛑 Shutting Down
To stop both the API and UI processes, simply press `Ctrl+C` in the terminal running `./run.sh`. The script includes a trap to ensure all background processes are terminated gracefully.
