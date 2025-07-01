DataFlow Pro
Automated Data Preprocessing & ML Workflow Assistant

![image](https://github.com/user-attachments/assets/8f8f0455-9112-4932-a8cc-978932d00b5f)


Overview
DataFlow Pro is an intuitive web application designed to streamline and automate the preliminary processes of data science workflows. It empowers both data scientists and non-data scientists to quickly profile, preprocess, feature engineer, and even build basic machine learning models on their datasets, all through an interactive user interface.
The core idea is to eliminate repetitive manual tasks, providing intelligent insights and actionable steps to prepare data for analysis and modeling.

Key Features
CSV File Upload: Easily upload your datasets for immediate processing.

Comprehensive Data Profiling:
Overall data summary (rows, columns).
Detailed column information (data types, missing value counts and percentages).
Descriptive statistics for numerical features (mean, median, std, min, max, quartiles, skewness, kurtosis).
Outlier detection (IQR method) with counts and percentages.
Categorical value counts and distributions.
Correlation heatmap for numerical features.

Visualizations: Automatically generated histograms, box plots, and bar charts to visually understand data distributions and relationships.

Intelligent AI Insights (Powered by Gemini 2.0 Flash):
Generates a comprehensive textual analysis of your dataset.
Provides a step-by-step data preprocessing and feature engineering procedure tailored to your specific data, making it easy for even non-data scientists to follow.
Offers general recommendations, potential target variables for modeling, and suitable model types.

Extensive Preprocessing Options:
Missing Value Handling: Impute with Mean, Median, Mode, K-Nearest Neighbors (KNN), or Multiple Imputation by Chained Equations (MICE). Options to remove rows or columns with missing values.
Duplicate Removal: Clean datasets by eliminating duplicate rows.
Column Dropping: Easily remove unnecessary columns.
Feature Scaling: Apply StandardScaler, MinMaxScaler, or RobustScaler to numerical features.
Skewness Transformation: Correct skewed distributions using Log, Square Root, Box-Cox, or Yeo-Johnson transformations.
Outlier Handling: Winsorize (cap) or trim (remove) outliers based on quantiles.

Powerful Feature Engineering:
Encoding Categorical Features: One-Hot Encoding and Frequency Encoding.
Interaction Features: Create new features by multiplying two existing numerical columns.
Ratio Features: Generate new features by dividing two existing numerical columns.
Lagged Features: Create time-series lagged features, grouped by a specified column and ordered by time.

Model Building & Evaluation:
Train common machine learning models: Linear Regression, Logistic Regression, Decision Tree Classifier, Random Forest Regressor, K-Means Clustering.
Evaluate model performance with relevant metrics (MSE, R2, Accuracy, Precision, Recall, F1-Score, Silhouette Score).
Visualize model diagnostics (Residual Plots, Confusion Matrices, ROC Curves).

Undo/Redo Functionality: Experiment freely with transformations with the ability to revert or reapply steps.

Export Processed Data: Download the current state of your transformed dataset as a CSV.

Export Python Code: Generate a runnable Python script of all applied preprocessing and feature engineering steps for reproducibility.

Integrated Notepad: A simple notepad to jot down thoughts, observations, or plan your workflow directly within the application.

🛠️ Technology Stack
Frontend:
React: A JavaScript library for building user interfaces.
Tailwind CSS: A utility-first CSS framework for rapid UI development.

Backend:
Python: The core language for data processing.
Flask: A lightweight Python web framework for building the API.
Pandas: For powerful data manipulation and analysis.
NumPy: Fundamental package for numerical computing.
Scikit-learn: Comprehensive library for machine learning (preprocessing, modeling, evaluation).
Matplotlib & Seaborn: For generating static data visualizations.
Gunicorn: A production-ready WSGI HTTP server for deploying Flask.
python-dotenv: For managing environment variables locally.

Artificial Intelligence:
Google Gemini 2.0 Flash API: Integrated for generating intelligent data insights and step-by-step recommendations.

Deployment:
Vercel: For hosting the React frontend.
Render: For hosting the Python Flask backend.

⚙️ Local Setup Instructions
To run DataFlow Pro on your local machine, follow these steps:

Prerequisites
Python 3.8+ and pip installed.
Node.js 14+ and npm installed.
Git installed.
A Google Gemini API Key.

1. Clone the Repository
First, clone the DataFlow Pro repository to your local machine:

git clone https://github.com/Vis-3/DataFlowPro.git
cd DataFlowPro

2. Backend Setup
Navigate into the root of your cloned repository (where app.py, requirements.txt, and Procfile are located).

# You should already be in the DataFlowPro directory from the previous step
# cd DataFlowPro

a.  Create and Activate a Python Virtual Environment:
bash python3 -m venv venv source venv/bin/activate 

b.  Install Python Dependencies:
bash pip install -r requirements.txt 
(If requirements.txt is missing, create it with the following content and then run pip install -r requirements.txt):
Flask Flask-Cors pandas numpy matplotlib seaborn scikit-learn google-generativeai python-dotenv scipy gunicorn

c.  Configure Gemini API Key:
Create a file named .env in the same directory as app.py and add your Gemini API key:
GOOGLE_API_KEY='YOUR_GEMINI_API_KEY_HERE'
Replace YOUR_GEMINI_API_KEY_HERE with your actual Gemini API Key.

d.  Run the Flask Backend:
bash python app.py 
The backend will start on http://127.0.0.1:5000. Keep this terminal window open.

3. Frontend Setup
Open a new terminal window and navigate into the frontend directory:

cd /home/sanskar/autodata/frontend # Adjust path if different

a.  Install Node.js Dependencies:
bash npm install 

b.  Install and Configure Tailwind CSS:
bash npm install -D tailwindcss postcss autoprefixer npx tailwindcss init -p 
* Edit tailwind.config.js: Open the tailwind.config.js file (in the frontend directory) and update the content array:
javascript /** @type {import('tailwindcss').Config} */ module.exports = { content: [ "./src/**/*.{js,jsx,ts,tsx}", "./public/index.html", ], theme: { extend: {}, }, plugins: [], } 
* Edit src/index.css: Open frontend/src/index.css and replace its entire content with:
css @tailwind base; @tailwind components; @tailwind utilities; 

c.  Update Backend URL in Frontend Code:
Open frontend/src/App.js and ensure the BACKEND_URL constant is set to use environment variables (for local development, it will fall back to localhost):
javascript // In frontend/src/App.js const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://127.0.0.1:5000'; 

d.  Run the React Frontend:
bash npm start 
The frontend will open in your browser, usually at http://localhost:3000.

☁️ Deployment (currently uses free tier for frontend and backend which makes it slow and limits functionality, runs perfectly fine locally)
DataFlow Pro is designed for cloud deployment:

Frontend (React): Deployed to Vercel for static site hosting and continuous deployment.

Backend (Flask): Deployed to Render as a web service.

Important: When deploying, ensure you set the REACT_APP_BACKEND_URL environment variable in your Vercel project settings to the public URL of your deployed Render backend (e.g., https://your-backend-name.onrender.com).

💡 Future Enhancements
Advanced Model Tuning: Hyperparameter optimization, cross-validation.

More ML Models: Support for XGBoost, LightGBM, neural networks.

Interactive Plotting: Integrate Plotly.js or Bokeh.js for dynamic visualizations.

Scalability for Large Datasets: Implement chunking or distributed processing for very large files.

Custom Feature Creation: Allow users to define custom Python functions for feature engineering.

Data Persistence: Integrate Firebase Firestore for saving and loading user sessions (processed data, history).

User Authentication: Implement full user authentication beyond anonymous access.

🤝 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

✉️ Contact
Sanskar Srivastava: https://github.com/Vis-3/
