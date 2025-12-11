# Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Load datasets
telco = pd.read_csv("Telco-Customer-Churn.csv")
sales = pd.read_csv("sample_sales_data.csv")

# Data Integration
sales_agg = sales.groupby("Customer").agg(
    total_revenue=("Revenue", "sum"),
    purchase_count=("OrderDate", "count"),
    avg_order_value=("Revenue", "mean")
).reset_index()

# Match columns (string-based)
telco.rename(columns={"customerID": "Customer"}, inplace=True)
data_fused = pd.merge(telco, sales_agg, on="Customer", how="left")

# Preprocessing
# Convert categorical variables to numeric
le = LabelEncoder()
categorical_columns = data_fused.select_dtypes(include=['object']).columns
for column in categorical_columns:
    if column != 'Customer':  # Skip the Customer ID column
        data_fused[column] = le.fit_transform(data_fused[column].astype(str))

# Handle missing values
data_fused.fillna(0, inplace=True)

# Separate features and target
X = data_fused.drop(['Customer', 'Churn'], axis=1)
y = data_fused['Churn']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    results[name] = {
        'Accuracy': f"{accuracy:.0%}",
        'F1': f"{f1:.2f}",
        'ROC-AUC': f"{auc:.2f}"
    }
    
    # Print results
    print(f"\n{name}:")
    print(f"Accuracy: {accuracy:.0%}")
    print(f"F1-Score: {f1:.2f}")
    print(f"ROC-AUC: {auc:.2f}")

# Create results DataFrame
results_df = pd.DataFrame(results).T
print("\nSummary Results:")
print(results_df)
