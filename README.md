# Project - Predicting Customer Churn using Data Integration & Fusion

## Background

- Businesses lose billions due to customer churn
- Churn prediction helps identify customers likely to leave
- Most churn models use limited internal data
- We aim to enhance churn prediction by integrating Telco churn data with sales transaction data

## Data Sources
Used following datasets for the project:

### 1.Dataset - Telco Customer Churn
- Description: Each row represents a customer who left and their demographic details
- Schema: Demographics, service usage, contract type, churn flag
- Summary Statistics
  - Unique values: 7043
  - Male & Female Ratio: 1:1

### 2.Dataset - Sample Sales Data
- Description: Represents total orders and sales of product categories across regions
- Schema: Sales orders, revenue, region, product category
- Summary Statistics
  - Unique values: 2823

## Data Integration & Fusion Techniques
- Applied schema matching and mapping to align fields (e.g., customer ID, region)
- Used string matching (Jaro-Winkler) for aligning customer names
- Performed data fusion to create unified customer profiles
- Merged sales metrics (total revenue, frequency) into churn dataset
