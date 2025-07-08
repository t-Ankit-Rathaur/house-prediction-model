## House Price Prediction Model Overview
- An end-to-end regression model using Python, Streamlit, and scikit-learn to estimate housing prices. Built for modularity, maintainability, and scalability.

### Project Structure
<pre>
HOUSE-PRICE-PREDICTION-MODEL/
├── .venv/                 # Virtual environment created via uv
├── .gitignore             # Git exclusions
├── .python-version        # Python version pin
├── pyproject.toml         # Dependency metadata
├── README.md              # Project guide
│
├── data/
│   ├── raw_data/          # Original datasets
│   └── processed_data/    # Cleaned datasets
│
├── config/
│   └── settings.yaml      # Centralized configuration file
│
├── logs/
│   └── app.log            # Runtime logs
│
├── notebooks/
│   └── exploration.ipynb  # EDA and experimentation
│
└── src/
    └── main.py            # Streamlit app and ML pipeline
</pre>

## Quickstart

# Clone the repository
git clone https://github.com/yourusername/house-prediction-model.git
cd house-prediction-model

# Install dependencies using uv
uv pip install

# Launch Streamlit app
streamlit run src/main.py

