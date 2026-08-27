# RevenueRescue AI

AI-powered revenue recovery decision system for the Razorpay AI Interns Buildathon.

## What it does

1. Identifies failed payments.
2. Predicts recovery probability using a Random Forest model.
3. Recommends a recovery action.
4. Explains the recommendation.
5. Estimates expected recovered revenue.
6. Provides an interactive Streamlit dashboard.

## Run locally

```bash
pip install -r requirements.txt
python generate_data.py
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## Important

This prototype uses synthetic data for demonstration. Do not connect it to real payment/customer data for the buildathon demo unless you have appropriate authorization.
