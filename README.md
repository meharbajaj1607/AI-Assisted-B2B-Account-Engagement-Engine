# AI-Assisted B2B Account Engagement Engine

## About the Project

I am building an AI-powered B2B Account Engagement Engine that helps businesses identify potential customers, prioritize accounts and generate personalized outreach.

The idea is to combine Machine Learning and Generative AI to make the account research and prioritization process faster and more organized.

Instead of manually going through every account, the system will:
- Predict the probability of an account converting into a customer.
- Prioritize accounts based on their conversion probability and engagement.
- Recommend the next best action for each account.
- Generate account insights and personalized outreach emails using Gemini.
- Allow a human to review and approve the generated emails before using them.

## How the System Works

```text
B2B Account Dataset
        |
        v
Data Preprocessing
        |
        v
ML Model
        |
        v
Conversion Probability
        |
        v
Next Best Action Engine
        |
        v
Gemini AI
        |
        v
Account Insights + Outreach Email
        |
        v
Flask REST API
        |
        v
React Dashboard
        |
        v
Human Review and Approval
```

## Tech Stack

- **Python** – Main programming language
- **Pandas & NumPy** – Data processing
- **Scikit-learn** – Machine Learning
- **Flask** – Backend and REST APIs
- **Google Gemini API** – AI-generated insights and outreach
- **Joblib** – Saving trained ML models
- **React + Vite** – Frontend dashboard (in progress)

## What I Have Built So Far

### 1. Dataset Generation

Created a synthetic B2B account dataset containing 1,000 accounts.

The dataset includes information such as:
- Company industry
- Employee count and annual revenue
- Technology match
- Website visits and email opens
- Content downloads
- Event attendance
- Demo requests
- Days since last activity
- Conversion status

### 2. Exploratory Data Analysis

Explored the dataset to understand which factors are associated with account conversion.

Some observations:
- Demo requests showed a strong positive correlation with conversion.
- Content downloads and website visits also showed positive correlations.
- More days since the last activity showed a negative correlation with conversion.

Also checked missing values, conversion distribution and industry-wise conversion rates.

### 3. Data Preprocessing

- Removed unnecessary columns.
- Converted categorical industry data using one-hot encoding.
- Split the dataset into training and testing sets using an 80:20 split.
- Saved the processed datasets for model training.

### 4. Machine Learning Model

Trained and compared three classification algorithms:

- Logistic Regression
- Random Forest
- Gradient Boosting

Model performance:

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 76.5% | 0.8306 |
| Random Forest | 77.0% | 0.8301 |
| Gradient Boosting | 75.5% | **0.8324** |

Selected Gradient Boosting based on its ROC-AUC score.

The model predicts the probability of an account converting into a customer.

*Note: These results are based on synthetic data and are for project demonstration purposes.*

### 5. Next Best Action Engine

After predicting the conversion probability, I implemented a rule-based engine to recommend what should be done next for each account.

For example:
- If an account has requested a demo, it gets high priority and an immediate sales follow-up recommendation.
- If conversion probability is high, the system recommends scheduling a sales call.
- If an account shows good engagement, it may be recommended for a personalized case study.
- If an account has been inactive for a long time, the system can recommend a re-engagement email.

This helps convert ML predictions into actionable recommendations.

### 6. Gemini AI Integration

Integrated the Google Gemini API to generate account-specific insights.

Based on the account information, ML prediction and NBA recommendation, Gemini generates:
- Account summaries
- Reasons why an account should be prioritized
- Personalized outreach emails

The ML model handles conversion prediction, the NBA engine decides the recommended action, and Gemini generates the explanations and communication.

### 7. Flask Backend

Created a Flask backend to connect all the components.

Implemented APIs for:
- Checking backend health.
- Retrieving all scored accounts.
- Retrieving individual account details.
- Generating AI-based account insights.
- Processing approval or rejection decisions.

The backend connects the dataset, ML model, NBA engine and Gemini service.

## Project Structure

```text
AI-Assisted-B2B-Account-Engagement-Engine/
│
├── backend/
│   ├── app.py
│   ├── generate_dataset.py
│   ├── explore_dataset.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── next_best_action.py
│   ├── gemini_service.py
│   ├── requirements.txt
│   └── models/
│
├── frontend/
│
├── data/
│   ├── raw/
│   │   └── b2b_accounts.csv
│   └── processed/
│
├── docs/
├── experiments/
├── .gitignore
└── README.md
```

## What's Next?

I am currently working on completing the remaining parts of the project.

1. **Backend testing:** Test all APIs and verify that the complete backend workflow works correctly.
2. **React Dashboard:** Build a dashboard to display accounts, conversion probabilities, priorities and recommended actions.
3. **Account Insights:** Display Gemini-generated summaries and prioritization reasons.
4. **Outreach Management:** Allow users to view, edit, approve or reject AI-generated emails.
5. **End-to-End Testing:** Test the complete workflow from account data to final outreach approval.

## Current Project Status

- [x] Project setup and GitHub repository
- [x] Synthetic dataset generation
- [x] Exploratory Data Analysis
- [x] Data preprocessing
- [x] ML model training and evaluation
- [x] Next Best Action Engine
- [x] Gemini API integration
- [x] Flask API implementation
- [ ] Backend API testing
- [ ] React dashboard
- [ ] Human approval workflow
- [ ] End-to-end testing

## Future Improvements

Some features I would like to explore after completing the MVP:
- Database integration to store account information and approval history.
- Slack integration for notifications.
- Confluence integration for accessing additional account information.
- Testing the model on real-world B2B account data.