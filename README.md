# Loan Approval Risk Model

A machine learning classification project that predicts loan approval decisions using applicant financial, credit, employment, and loan information.

The project follows the **CRISP-DM framework** and evaluates multiple classification models while considering both predictive performance and the financial consequences of incorrect lending decisions.

## Business Problem

Loan approval processes can be time-consuming and inconsistent when they rely heavily on manual assessment.

This project explores how machine learning could support loan officers by providing a more consistent and data-driven approach to evaluating loan applications.

Two types of prediction errors have different assumed financial consequences:

- **False Positive:** Approving a loan that should not be approved — estimated cost of **$50,000**
- **False Negative:** Rejecting a creditworthy applicant — estimated opportunity cost of **$8,000**

Because a false approval is considerably more expensive, model evaluation considers not only traditional classification metrics but also the estimated financial impact of prediction errors.

## Dataset

The analysis uses a dataset containing **20,000 loan applications** and **35 variables** describing applicant demographics, financial position, credit history, employment, and loan characteristics.

The target variable is:

- `LoanApproved = 1` — Approved
- `LoanApproved = 0` — Denied

The target is imbalanced, with approximately **24% approved** and **76% denied** applications.

### Important Dataset Limitation

The supplied data-generation process indicates that the dataset is **synthetic**. The loan approval outcome is generated using predefined relationships between applicant and loan characteristics.

The project should therefore be viewed as a demonstration of an end-to-end machine learning workflow rather than evidence of expected performance on real-world lending data.

The target also represents **historical loan approval**, not actual future loan default.

## Project Workflow

The project follows the CRISP-DM methodology:

1. **Business Understanding**
   - Defined the loan approval problem
   - Identified stakeholders
   - Evaluated the financial consequences of classification errors
   - Established model evaluation criteria

2. **Data Understanding**
   - Explored feature distributions
   - Investigated missing values and class imbalance
   - Examined feature-target relationships
   - Identified potential data leakage

3. **Data Preparation**
   - Converted currency-formatted variables to numerical values
   - Removed the leaking `RiskScore` feature
   - Imputed missing numerical and categorical values
   - One-hot encoded categorical features
   - Standardized numerical features
   - Created reusable Scikit-learn preprocessing pipelines

4. **Modeling**
   - Dummy Classifier
   - Logistic Regression
   - Decision Tree
   - Random Forest

5. **Optimization**
   - GridSearchCV
   - Regularization tuning
   - Class-weight comparison
   - Cross-validation using F1-score

6. **Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
   - Confusion Matrix
   - Estimated business cost
   - Segment-level performance
   - Feature importance

## Data Leakage Investigation

Exploratory analysis identified `RiskScore` as having an unusually strong relationship with loan approval.

Review of the supplied data-generation logic showed that `RiskScore` was calculated after `LoanApproved` and directly adjusted using the approval outcome.

Including this feature would therefore introduce **target leakage** and artificially inflate model performance.

`RiskScore` was consequently removed before model training.

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Estimated Business Cost |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 96.5% | 92.9% | 92.4% | 92.7% | 0.995 | $3.934M |
| Decision Tree | 88.7% | 75.8% | 77.4% | 76.6% | 0.848 | $13.528M |
| Random Forest | 93.2% | 90.0% | 80.8% | 85.1% | 0.978 | $5.772M |

Logistic Regression provided the strongest initial combination of predictive performance, interpretability, and estimated financial impact.

## Model Optimization

The Logistic Regression model was optimized using `GridSearchCV`.

The parameter search evaluated different combinations of:

- Regularization strength (`C`)
- Class weighting

The best configuration was:

```python
LogisticRegression(
    C=10,
    class_weight=None,
    max_iter=1000,
    random_state=42
)
```

The best mean cross-validation F1-score was approximately **0.922**.

## Final Model Performance

The optimized Logistic Regression model achieved:

| Metric | Result |
|---|---:|
| Accuracy | **96.6%** |
| Precision | **93.0%** |
| Recall | **92.7%** |
| F1-Score | **92.8%** |
| ROC-AUC | **0.995** |
| False Positives | **67** |
| False Negatives | **70** |
| Estimated Business Cost | **$3.91M** |

Optimization reduced false negatives from 73 to 70 while keeping false positives at 67, reducing the estimated business cost by approximately **$24,000** compared with the original Logistic Regression model.

## Key Predictive Features

Analysis of the optimized Logistic Regression coefficients identified several influential predictors.

Important negative associations with approval included:

- Total debt-to-income ratio
- Bankruptcy history
- Interest rate
- Loan amount
- Unemployment status

Important positive associations included:

- Monthly income
- Net worth
- Length of credit history

These relationships describe how the model makes predictions and should **not be interpreted as causal relationships**.

## Business Recommendations

The model demonstrates how machine learning could support more consistent and data-driven loan assessment.

The main recommendations are:

- Use the model as a **decision-support tool** rather than a fully autonomous approval system.
- Pay particular attention to applicant affordability measures such as debt relative to income, income, loan amount, and overall financial position.
- Refer uncertain or high-risk applications for additional human review.
- Monitor false approvals carefully because their assumed financial impact is substantially greater than that of false denials.
- Monitor model performance across different applicant groups.
- Review the appropriateness of variables such as education, marital status, employment status, and home ownership before real-world use.

## Limitations

The project has several important limitations:

- The dataset is synthetically generated.
- The target represents historical approval rather than actual loan repayment/default.
- Synthetic relationships may be cleaner than those found in real financial data.
- The model has not been externally validated on real lending data.
- Similar overall accuracy across applicant groups does not by itself establish model fairness.

The reported model performance should therefore not be assumed to represent expected performance in a real lending environment.

## Future Improvements

Future development could include:

- Training and validating the model using real loan repayment and default data
- Performing more comprehensive fairness analysis
- Optimizing the classification threshold according to business costs
- Monitoring model and data drift
- Testing additional feature-engineering strategies
- Implementing model monitoring and periodic retraining
- Establishing formal human-review and model-governance procedures

## Repository Structure

```text
loan-approval-risk-model/
│
├── data/
│   └── financial_loan_data.csv
│
├── notebooks/
│   └── loan_approval_model.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Git & GitHub

## Running the Project

Clone the repository:

```bash
git clone https://github.com/ngunjirikevin/financial-loan-risk-model
cd loan-approval-risk-model
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Then open:

```text
notebooks/loan_approval_model.ipynb
```

and run the notebook cells in order.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for loan approval classification, covering business understanding, exploratory data analysis, preprocessing, leakage detection, model development, hyperparameter optimization, evaluation, and model interpretation.

Among the evaluated models, the optimized Logistic Regression model provided the strongest combination of predictive performance, interpretability, and estimated financial impact.

While the results demonstrate the potential of machine learning for lending decision support, validation using representative real-world repayment data, fairness assessment, model monitoring, and appropriate human oversight would be necessary before operational deployment.