# Loan Approval Prediction System using Machine Learning

## Overview

This project is a Machine Learning-based Loan Approval Prediction System that predicts whether a loan application is likely to be approved based on applicant information such as income, education, credit history, loan amount, and other factors.

The goal of this project is to automate the loan approval process and assist financial institutions in making faster and more accurate decisions.

---

## Features

- Data preprocessing and cleaning
- Handling missing values
- Feature engineering
- Exploratory Data Analysis (EDA)
- Machine Learning model training
- Loan approval prediction
- Model evaluation and performance analysis
- User-friendly prediction workflow

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Jupyter Notebook

---

## Project Structure

```
Loan_approval_system/
│
├── app.py
├── Train.csv
├── Test_new.csv
├── loan_project_final.ipynb
├── Testing_unseendata.ipynb
├── loan_project_final.pkl
├── requirements.txt
└── result.csv
```

---

## Dataset Features

The model uses the following features:

- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area

Target Variable:

- Loan Status (Approved / Rejected)

---

## Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Missing Value Handling
4. Feature Encoding
5. Model Training
6. Model Evaluation
7. Loan Approval Prediction

---

## Installation

Clone the repository:

```bash
git clone https://github.com/vishwa024/Loan_approval_system.git
```

Navigate to the project folder:

```bash
cd Loan_approval_system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

---

## Results

The model was trained using historical loan application data and evaluated using standard machine learning metrics.

Key objectives achieved:

- Automated loan approval prediction
- Improved decision-making process
- Reduced manual analysis effort

---

## Future Improvements

- Deploy using Streamlit
- Add Flask/Django web interface
- Improve model accuracy with ensemble methods
- Integrate real-time prediction API
- Add user authentication

---

## Author

**Vishwa Mistry**

GitHub: https://github.com/vishwa024

LinkedIn: https://linkedin.com/in/vishwa-mistry-932a49368
