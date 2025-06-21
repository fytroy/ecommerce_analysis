# E-Commerce Data Analysis and Customer Segmentation

## Description

This project provides a Python script (`analyze.py`) for comprehensive e-commerce data analysis. It processes transactional data to uncover insights into sales trends, product performance, customer behavior, and geographic sales distribution. The script can generate its own synthetic dataset for demonstration purposes or can be adapted to use real-world e-commerce data.

Key analyses performed include:
*   **Data Simulation:** Generates a realistic, large-scale e-commerce dataset.
*   **Data Cleaning & Preparation:** Ensures data quality through robust cleaning procedures.
*   **Exploratory Data Analysis (EDA):** Visualizes sales trends, product performance, customer spending habits, geographic sales distribution, and payment method preferences.
*   **RFM (Recency, Frequency, Monetary) Analysis:** Segments customers based on their purchasing behavior to identify key customer groups.

## Features

*   **Synthetic Data Generation:** Creates a sample dataset of ~250,000 e-commerce transactions.
*   **Detailed Data Cleaning:** Includes handling of missing values, data type conversions, standardization of categorical data, and removal of duplicates or erroneous entries.
*   **Comprehensive EDA:**
    *   Sales trends over time (monthly, daily, hourly).
    *   Top product performance (by revenue and quantity).
    *   Sales distribution by product category.
    *   Customer spending analysis (Average Order Value, distribution of total spend).
    *   Geographic sales analysis (by shipping city).
    *   Payment method popularity.
*   **Customer Segmentation using RFM:**
    *   Calculates Recency, Frequency, and Monetary scores for each customer.
    *   Segments customers into meaningful categories (e.g., "Champions", "Loyal Customers", "Lost Customers").
    *   Visualizes segment distribution.

## Technologies Used

*   Python 3.x
*   pandas
*   numpy
*   matplotlib
*   seaborn

## Setup and Installation

1.  **Clone the repository (if applicable) or download the `analyze.py` script.**
2.  **Ensure you have Python 3 installed.** You can download it from [python.org](https://www.python.org/).
3.  **Install the required libraries.** It's recommended to use a virtual environment.
    Open your terminal or command prompt and run:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

Navigate to the directory containing the `analyze.py` script and run it from your terminal:

```bash
python analyze.py
```

## Output

The script will:
1.  Print status messages to the console as it progresses through different stages of analysis (data simulation, cleaning, EDA, RFM).
2.  Display several plot windows sequentially for each visualization generated during the EDA and RFM analysis. You will need to close each plot window to allow the script to proceed to the next visualization.
3.  Print summaries and results, such as descriptive statistics, RFM segment distributions, and average RFM values per segment, to the console.

## Project Structure

```
.
├── analyze.py       # Main Python script for data analysis
└── README.md        # This file
```
(Other files like `.idea/` for IDE configuration may also be present.)
