# app.py
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Scikit-learn imports for preprocessing
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer # Explicitly enable experimental IterativeImputer
from sklearn.impute import IterativeImputer # Now import IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer # For Yeo-Johnson

# Scikit-learn imports for model building and evaluation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, silhouette_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress ConvergenceWarning from scikit-learn
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning) # Suppress KMeans n_init warning


app = Flask(__name__)
CORS(app)

current_dataframe = None
original_dataframe = None # This will store the initial dataframe upon upload, used for 'reset all'

# Configure the Gemini API client. Get the API key from environment variable.
# IMPORTANT: For local development, you MUST set GOOGLE_API_KEY in your WSL terminal:
# export GOOGLE_API_KEY='YOUR_GEMINI_API_KEY_HERE'
 # Placeholder for Canvas. Will be populated at runtime.
genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
@app.route('/')
def hello_world():
    """A simple test route to ensure the backend is running."""
    return 'Hello from the Python Backend!'

def generate_plot_image(fig):
    """
    Saves a matplotlib Figure object to a BytesIO object and converts it to a base64 string.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return image_base64

def replace_nan_with_none(obj):
    """Recursively replaces np.nan with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_none(item) for item in obj] 
    elif pd.isna(obj):
        return None
    return obj

def calculate_outliers_iqr(series):
    """Calculates outliers using the IQR method for a numerical series."""
    if pd.api.types.is_numeric_dtype(series) and series.count() > 0:
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return {
            'count': int(outliers.count()),
            'percentage': round((outliers.count() / series.count() * 100), 2) if series.count() > 0 else 0.0,
            'lower_bound': round(lower_bound, 2),
            'upper_bound': round(upper_bound, 2)
        }
    return None

def perform_profiling_and_plotting(df):
    """
    Performs comprehensive data profiling and generates plots for the given DataFrame.
    Returns a dictionary containing all summary data, plots, and explanations.
    """
    num_rows = df.shape[0]
    num_cols = df.shape[1]
    column_names = df.columns.tolist()
    data_types = df.dtypes.astype(str).to_dict()
    missing_values = df.isnull().sum().to_dict()
    missing_percentages = (df.isnull().sum() / num_rows * 100).round(2).to_dict() if num_rows > 0 else {col: 0.0 for col in column_names}

    # Identify numerical and categorical columns early
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Calculate descriptive statistics including skewness, kurtosis, and additional quantiles
    quantiles_to_include = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    descriptive_stats_df = df.describe(include='all', percentiles=quantiles_to_include)

    numerical_df = df.select_dtypes(include=np.number)
    
    # Calculate skewness info explicitly
    skewness_info = {}
    if not numerical_df.empty:
        skew_series = numerical_df.skew()
        for col in numerical_cols:
            skewness_info[col] = round(skew_series.get(col, np.nan), 2) if not pd.isna(skew_series.get(col)) else None

    if not numerical_df.empty:
        descriptive_stats_df.loc['skew'] = numerical_df.skew()
        descriptive_stats_df.loc['kurtosis'] = numerical_df.kurtosis()
    descriptive_stats = replace_nan_with_none(descriptive_stats_df.to_dict(orient='index'))

    head_data = df.head().to_json(orient='split', index=False)
    tail_data = df.tail().to_json(orient='split', index=False)

    plots = {}
    column_details = {}
    outlier_info = {}

    # Calculate Outlier Info
    for col in numerical_cols:
        outlier_info[col] = calculate_outliers_iqr(df[col])


    # Unique Value Counts and Top Values for Categorical Columns
    for col in categorical_cols:
        value_counts = df[col].value_counts(dropna=False)
        total_non_missing = df[col].count()
        top_n = 10
        column_details[col] = {
            'unique_count': df[col].nunique(dropna=False),
            'value_counts': value_counts.head(top_n).to_dict(),
            'value_percentages': (value_counts.head(top_n) / total_non_missing * 100).round(2).to_dict() if total_non_missing > 0 else {val: 0.0 for val in value_counts.head(top_n).index},
            'has_more_values': df[col].nunique(dropna=False) > top_n
        }

    # Histograms for numerical columns
    for col in numerical_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=20, ax=ax)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        plots[f'hist_{col}'] = generate_plot_image(fig)

    # Box Plots for numerical columns
    for col in numerical_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=df[col].dropna(), ax=ax)
        ax.set_title(f'Box Plot of {col}')
        ax.set_xlabel(col)
        plots[f'box_{col}'] = generate_plot_image(fig)

    # Bar Plots for categorical columns (top N values)
    for col in categorical_cols:
        top_values = df[col].value_counts().head(10)
        if not top_values.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=top_values.index, y=top_values.values, ax=ax)
            ax.set_title(f'Top Categories for {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plots[f'bar_{col}'] = generate_plot_image(fig)
        else:
            plots[f'bar_{col}'] = None

    # Correlation Heatmap
    correlation_matrix_data = None
    if len(numerical_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
        ax.set_title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plots['correlation_heatmap'] = generate_plot_image(fig)
        correlation_matrix_data = replace_nan_with_none(corr_matrix.to_dict(orient='records'))
    else:
        plots['correlation_heatmap'] = None

    # Explanations Section
    explanations = {
        'overall_overview': {
            'title': 'Overall Data Overview',
            'description': (
                "This section provides a high-level summary of your dataset. "
                "Understanding the number of rows and columns, along with initial data types, "
                "is fundamental before proceeding with any in-depth analysis or preprocessing."
            )
        },
        'column_details': {
            'title': 'Column Details',
            'description': (
                "This table lists each column, its inferred data type, and the count and percentage of missing values. "
                "It's crucial to review data types to ensure they are correct (e.g., numbers aren't treated as text). "
                "High percentages of missing values ('Missing %') indicate columns that might need specific handling, "
                "such as imputation (filling in missing data) or removal if too much data is absent."
            )
        },
        'descriptive_statistics': {
            'title': 'Descriptive Statistics (Numerical & Categorical)',
            'description': (
                "For numerical columns, this table provides key metrics like count, mean (average), standard deviation (spread), "
                "minimum, maximum, and quartiles (25th, 50th/median, 75th percentiles). Additional quantiles (1st, 5th, 95th, 99th) "
                "offer more insight into the tails of the distribution and extreme values. "
                "Skewness (asymmetry) and Kurtosis (tailedness) are also included, indicating the shape of the distribution. "
                "Vastly different scales (min/max) across numerical features often require feature scaling (e.g., Min-Max or Standard Scaling) "
                "for many machine learning algorithms to perform optimally. For categorical columns, it shows unique counts and top occurrences."
            )
        },
        'skewness_information': {
            'title': 'Skewness Information (Numerical Features)',
            'description': (
                "Skewness measures the asymmetry of the probability distribution of a real-valued random variable about its mean. "
                "A positive skew indicates a tail on the right side of the distribution, while a negative skew indicates a tail on the left side. "
                "Values close to 0 suggest a symmetrical distribution. "
                "**Interpretation:** Highly skewed distributions can negatively impact the performance of some machine learning models that assume normality (e.g., linear regression). "
                "**Suggestion:** Skewed data often benefits from transformations (e.g., logarithmic, square root, Box-Cox, Yeo-Johnson) to make them more symmetrical and Gaussian-like."
            )
        },
        'outlier_information': {
            'title': 'Outlier Information (IQR Method)',
            'description': (
                "This section quantifies potential outliers in numerical columns using the Interquartile Range (IQR) method. "
                "Outliers are data points that fall below $Q1 - 1.5 \\times IQR$ or above $Q3 + 1.5 \\times IQR$. "
                "**Interpretation:** A high count or percentage of outliers might indicate data quality issues, measurement errors, or genuinely extreme but important values. "
                "**Suggestion:** Decide on an outlier treatment strategy: remove, cap (winsorize), or use robust scaling techniques that are less sensitive to outliers. The best approach depends on the nature of the data and the problem."
            )
        },
        'categorical_column_details': {
            'title': 'Categorical Column Value Counts',
            'description': (
                "This section provides a detailed breakdown of unique values for each categorical column, along with their counts and percentages. "
                "Look for misspellings, inconsistent entries (e.g., 'USA' vs 'U.S.A.'), or categories with very low counts that might need grouping. "
                "Columns with a very high number of unique values (high cardinality) may require specific encoding strategies "
                "(e.g., target encoding, frequency encoding) or careful consideration before one-hot encoding, as this can lead to a very wide dataset."
            )
        },
        'visualizations': {
            'title': 'Visualizations',
            'description': (
                "Visualizations provide an intuitive way to understand data distributions, relationships, and identify issues that raw numbers might miss. "
                "They are crucial for preprocessing decisions and selecting appropriate models."
            )
        },
        'correlation_heatmap': {
            'title': 'Correlation Heatmap (Numerical Features)',
            'description': (
                "This heatmap shows the pairwise correlation between numerical features. Values closer to 1 or -1 indicate a strong positive or negative linear relationship, respectively. "
                "Values near 0 suggest little to no linear relationship. "
                "**Interpretation:** High correlations (e.g., > 0.8 or < -0.8) might indicate multicollinearity, which can be problematic for linear models. "
                "**Suggestion:** For highly correlated features, consider keeping only one, combining them, or using dimensionality reduction techniques like PCA."
            )
        },
        'numerical_distributions_outliers': {
            'title': 'Numerical Feature Distributions & Outliers',
            'description': (
                "**Histograms** display the frequency distribution of numerical features. Look for skewness (data clustered to one side), multiple peaks (modes), or unusual shapes. "
                "**Box Plots** summarize the distribution using quartiles and visually highlight outliers (points beyond the 'whiskers'). "
                "**Interpretation:** Highly skewed distributions may benefit from transformations (log, square root) to make them more symmetrical, which is often preferred by models assuming normality. "
                "Outliers identified in box plots might be data entry errors or genuine extreme values. "
                "**Suggestion:** Decide on an outlier treatment strategy: remove, cap (winsorize), or use robust scaling techniques that are less sensitive to outliers."
            )
        },
        'categorical_distributions': {
            'title': 'Categorical Feature Distributions',
            'description': (
                "These bar plots show the frequency of each category within your categorical features. "
                "**Interpretation:** Observe if categories are highly imbalanced (one category has significantly more or fewer observations than others). "
                "This can impact model training, especially for classification tasks where the target variable is imbalanced. "
                "**Suggestion:** For imbalanced categories, consider techniques like oversampling, undersampling, or generating synthetic samples (e.g., SMOTE) if the feature is a target variable. "
                "For feature columns, highly imbalanced categories might be less informative."
            )
        }
    }

    return {
        "data_summary": {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "column_names": column_names,
            "data_types": data_types,
            "missing_values": missing_values,
            "missing_percentages": missing_percentages
        },
        "data_preview_head": head_data,
        "data_preview_tail": tail_data,
        "descriptive_statistics": descriptive_stats,
        "outlier_info": outlier_info,
        "skewness_info": skewness_info,
        "correlation_matrix_data": correlation_matrix_data,
        "plots": plots,
        "column_details": column_details,
        "explanations": explanations
    }


@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """
    Handles CSV file uploads from the frontend.
    Reads the CSV into a pandas DataFrame and performs initial profiling.
    """
    global current_dataframe, original_dataframe

    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        try:
            csv_data = io.StringIO(file.stream.read().decode('utf-8'))
            df = pd.read_csv(csv_data)
            current_dataframe = df.copy()
            original_dataframe = df.copy() # Store the original for reset functionality

            profiling_results = perform_profiling_and_plotting(current_dataframe)

            return jsonify({
                "status": "success",
                "message": f"CSV '{file.filename}' uploaded and processed successfully!",
                **profiling_results
            }), 200

        except Exception as e:
            print(f"Error processing CSV: {e}")
            return jsonify({"status": "error", "message": f"Error processing CSV: {str(e)}"}), 500
    else:
        return jsonify({"status": "error", "message": "Invalid file type. Please upload a CSV file."}), 400

# --- Preprocessing Endpoints ---

@app.route('/preprocess/impute_missing', methods=['POST'])
def impute_missing():
    """
    Imputes missing values in a specified column(s) using the chosen strategy.
    Supports mean, median, mode, KNN, and MICE imputation.
    """
    global current_dataframe

    if current_dataframe is None:
        return jsonify({"status": "error", "message": "No data uploaded. Please upload a CSV first."}), 400

    data = request.json
    column = data.get('column') # Can be a single column or 'all_numerical' / 'all_categorical'
    strategy = data.get('strategy') # 'mean', 'median', 'mode', 'knn', 'mice'
    k_neighbors = data.get('k_neighbors', 5) # For KNN

    if not strategy:
        return jsonify({"status": "error", "message": "Missing 'strategy' parameter."}), 400

    target_columns = []
    if column == 'all_numerical':
        target_columns = current_dataframe.select_dtypes(include=np.number).columns.tolist()
    elif column == 'all_categorical':
        target_columns = current_dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    elif column: # Specific column
        if column not in current_dataframe.columns:
            return jsonify({"status": "error", "message": f"Column '{column}' not found."}), 400
        target_columns = [column]
    else:
        return jsonify({"status": "error", "message": "Missing 'column' or invalid value."}), 400

    if not target_columns:
        return jsonify({"status": "success", "message": "No relevant columns to impute for the selected option.", **perform_profiling_and_plotting(current_dataframe)}), 200

    try:
        if strategy in ['mean', 'median', 'mode']:
            for col in target_columns:
                if current_dataframe[col].isnull().sum() > 0:
                    if strategy == 'mean' and np.issubdtype(current_dataframe[col].dtype, np.number):
                        current_dataframe[col].fillna(current_dataframe[col].mean(), inplace=True)
                    elif strategy == 'median' and np.issubdtype(current_dataframe[col].dtype, np.number):
                        current_dataframe[col].fillna(current_dataframe[col].median(), inplace=True)
                    elif strategy == 'mode':
                        # Mode can return multiple values, take the first one
                        mode_val = current_dataframe[col].mode()[0]
                        current_dataframe[col].fillna(mode_val, inplace=True)
                    else:
                        return jsonify({"status": "error", "message": f"Strategy '{strategy}' is not suitable for column '{col}' of type '{current_dataframe[col].dtype}'."}), 400
            message = f"Missing values in selected columns imputed with {strategy}."

        elif strategy == 'knn':
            numerical_cols_with_missing = [col for col in target_columns if np.issubdtype(current_dataframe[col].dtype, np.number) and current_dataframe[col].isnull().sum() > 0]
            if not numerical_cols_with_missing:
                 return jsonify({"status": "success", "message": "No numerical columns with missing values to impute using KNN.", **perform_profiling_and_plotting(current_dataframe)}), 200

            knn_imputer = KNNImputer(n_neighbors=int(k_neighbors))
            # Create a copy for imputation to avoid settingwithcopywarning and ensure original isn't modified by accident
            df_numerical_for_imputation = current_dataframe[numerical_cols_with_missing].copy()
            current_dataframe[numerical_cols_with_missing] = knn_imputer.fit_transform(df_numerical_for_imputation)
            message = f"Missing values in numerical columns imputed using KNN (k={k_neighbors})."
        
        elif strategy == 'mice':
            numerical_cols_with_missing = [col for col in target_columns if np.issubdtype(current_dataframe[col].dtype, np.number) and current_dataframe[col].isnull().sum() > 0]
            if not numerical_cols_with_missing:
                return jsonify({"status": "success", "message": "No numerical columns with missing values to impute using MICE.", **perform_profiling_and_plotting(current_dataframe)}), 200

            mice_imputer = IterativeImputer(max_iter=10, random_state=0) # Use a fixed random_state for reproducibility
            df_numerical_for_imputation = current_dataframe[numerical_cols_with_missing].copy()
            current_dataframe[numerical_cols_with_missing] = mice_imputer.fit_transform(df_numerical_for_imputation)
            message = f"Missing values in numerical columns imputed using MICE."

        else:
            return jsonify({"status": "error", "message": f"Invalid imputation strategy: '{strategy}'."}), 400

        profiling_results = perform_profiling_and_plotting(current_dataframe)
        return jsonify({"status": "success", "message": message, **profiling_results}), 200

    except Exception as e:
        print(f"Error imputing missing values: {e}")
        return jsonify({"status": "error", "message": f"Error imputing missing values: {str(e)}"}), 500

@app.route('/preprocess/remove_missing_rows', methods=['POST'])
def remove_missing_rows():
    """
    Removes rows that contain any missing values.
    """
    global current_dataframe

    if current_dataframe is None:
        return jsonify({"status": "error", "message": "No data uploaded. Please upload a CSV first."}), 400

    initial_rows = current_dataframe.shape[0]
    current_dataframe.dropna(inplace=True)
    rows_removed = initial_rows - current_dataframe.shape[0]

    message = f"Rows with missing values removed. {rows_removed} rows deleted."

    profiling_results = perform_profiling_and_plotting(current_dataframe)
    return jsonify({"status": "success", "message": message, **profiling_results}), 200

@app.route('/preprocess/remove_missing_cols', methods=['POST'])
def remove_missing_cols():
    """
    Removes columns that contain any missing values.
    """
    global current_dataframe

    if current_dataframe is None:
        return jsonify({"status": "error", "message": "No data uploaded. Please upload a CSV first."}), 400

    cols_before = set(current_dataframe.columns)
    current_dataframe.dropna(axis=1, inplace=True)
    cols_after = set(current_dataframe.columns)
    cols_removed = list(cols_before - cols_after)

    message = f"Columns with missing values removed. Columns deleted: {', '.join(cols_removed) if cols_removed else 'None'}."

    profiling_results = perform_profiling_and_plotting(current_dataframe)
    return jsonify({"status": "success", "message": message, **profiling_results}), 200

@app.route('/preprocess/remove_duplicates', methods=['POST'])
def remove_duplicates():
    """
    Removes duplicate rows from the DataFrame.
    """
    global current_dataframe

    if current_dataframe is None:
        return jsonify({"status": "error", "message": "No data uploaded. Please upload a CSV first."}), 400

    initial_rows = current_dataframe.shape[0]
    current_dataframe.drop_duplicates(inplace=True)
    rows_removed = initial_rows - current_dataframe.shape[0]

    message = f"Duplicate rows removed. {rows_removed} rows deleted."

    profiling_results = perform_profiling_and_plotting(current_dataframe)
    return jsonify({"status": "success", "message": message, **profiling_results}), 200

@app.route('/data/drop_column', methods=['POST'])
def drop_column():
    """
    Drops a specified column from the DataFrame.
    """
    global current_dataframe

    if current_dataframe is None:
        return jsonify({"status": "error", "message": "No data uploaded. Please upload a CSV first."}), 400

    data = request.json
    column_name = data.get('column_name')

    if not column_name:
        return jsonify({"status": "error", "message": "Missing 'column_name' parameter."}), 400

    if column_name not in current_dataframe.columns:
        return jsonify({"status": "error", "message": f"Column '{column_name}' not found."}), 400

    try:
        current_dataframe.drop(columns=[column_name], inplace=True)
        message = f"Column '{column_name}' dropped successfully."
        profiling_results = perform_profiling_and_plotting(current_dataframe)
        return jsonify({"status": "success", "message": message, **profiling_results}), 200
    except Exception as e:
        print(f"Error dropping column: {e}")
        return jsonify({"status": "error", "message": f"Error dropping column: {str(e)}"}), 500

@app.route('/transform/scale_features', methods=['POST'])
def scale_features():
    """
    Applies feature scaling (Standard, MinMax, Robust) to specified numerical columns.
    """
    global current_dataframe

    if current_dataframe is None:
        return jsonify({"status": "error", "message": "No data uploaded. Please upload a CSV first."}), 400

    data = request.json
    columns = data.get('columns', []) # List of columns to scale
    scaler_type = data.get('scaler_type') # 'standard', 'minmax', 'robust'

    if not columns or not scaler_type:
        return jsonify({"status": "error", "message": "Missing 'columns' or 'scaler_type' parameter."}), 400
    
    numerical_cols_to_scale = [col for col in columns if np.issubdtype(current_dataframe[col].dtype, np.number)]
    if not numerical_cols_to_scale:
        return jsonify({"status": "success", "message": "No numerical columns selected for scaling or found.", **perform_profiling_and_plotting(current_dataframe)}), 200

    try:
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            return jsonify({"status": "error", "message": f"Invalid scaler type: {scaler_type}."}), 400

        # Apply scaler only to selected numerical columns
        current_dataframe[numerical_cols_to_scale] = scaler.fit_transform(current_dataframe[numerical_cols_to_scale])
        message = f"Features in {', '.join(numerical_cols_to_scale)} scaled using {scaler_type} scaler."

        profiling_results = perform_profiling_and_plotting(current_dataframe)
        return jsonify({"status": "success", "message": message, **profiling_results}), 200
    except Exception as e:
        print(f"Error scaling features: {e}")
        return jsonify({"status": "error", "message": f"Error scaling features: {str(e)}"}), 500

@app.route('/transform/apply_skew_transform', methods=['POST'])
def apply_skew_transform():
    """
    Applies skewness transformation (Log, Sqrt, Box-Cox, Yeo-Johnson) to a specified numerical column.
    """
    global current_dataframe

    if current_dataframe is None:
        return jsonify({"status": "error", "message": "No data uploaded. Please upload a CSV first."}), 400

    data = request.json
    column_name = data.get('column_name')
    transform_type = data.get('transform_type') # 'log', 'sqrt', 'boxcox', 'yeo-johnson'

    if not column_name or not transform_type:
        return jsonify({"status": "error", "message": "Missing 'column_name' or 'transform_type' parameter."}), 400

    if column_name not in current_dataframe.columns:
        return jsonify({"status": "error", "message": f"Column '{column_name}' not found."}), 400

    if not np.issubdtype(current_dataframe[column_name].dtype, np.number):
        return jsonify({"status": "error", "message": f"Column '{column_name}' is not numerical. Cannot apply {transform_type}."}), 400

    try:
        # Before applying transformation, handle NaNs. Transformations will preserve NaNs but operate on valid data.
        # Use .copy() to avoid SettingWithCopyWarning if original column is a slice.
        col_series = current_dataframe[column_name].dropna().copy()

        if transform_type == 'log':
            # Use log1p to handle zero values gracefully by adding 1
            transformed_values = np.log1p(col_series)
            message = f"Log transformation applied to '{column_name}'."
        elif transform_type == 'sqrt':
            transformed_values = np.sqrt(col_series)
            message = f"Square root transformation applied to '{column_name}'."
        elif transform_type == 'boxcox':
            if (col_series <= 0).any():
                return jsonify({"status": "error", "message": f"Box-Cox transformation requires all values in '{column_name}' to be strictly positive after dropping NaNs."}), 400
            transformed_values, _ = boxcox(col_series)
            message = f"Box-Cox transformation applied to '{column_name}'."
        elif transform_type == 'yeo-johnson':
            pt = PowerTransformer(method='yeo-johnson')
            # Reshape for fit_transform, then flatten back
            transformed_values = pt.fit_transform(col_series.values.reshape(-1, 1)).flatten()
            message = f"Yeo-Johnson transformation applied to '{column_name}'."
        else:
            return jsonify({"status": "error", "message": f"Invalid transform type: {transform_type}."}), 400
        
        # Assign transformed values back, aligning by index to preserve NaNs
        current_dataframe[column_name] = transformed_values
        
        profiling_results = perform_profiling_and_plotting(current_dataframe)
        return jsonify({"status": "success", "message": message, **profiling_results}), 200
    except Exception as e:
        print(f"Error applying skew transformation: {e}")
        return jsonify({"status": "error", "message": f"Error applying {transform_type} transformation to '{column_name}': {str(e)}"}), 500

@app.route('/transform/one_hot_encode', methods=['POST'])
def one_hot_encode():
    """
    Applies one-hot encoding to a specified categorical column.
    """
    global current_dataframe

    if current_dataframe is None:
        return jsonify({"status": "error", "message": "No data uploaded. Please upload a CSV first."}), 400

    data = request.json
    column_name = data.get('column_name')

    if not column_name:
        return jsonify({"status": "error", "message": "Missing 'column_name' parameter."}), 400

    if column_name not in current_dataframe.columns:
        return jsonify({"status": "error", "message": f"Column '{column_name}' not found."}), 400
    
    # Check if the column is truly categorical or object type, not numerical
    if pd.to_numeric(current_dataframe[column_name], errors='coerce').notna().all() and current_dataframe[column_name].nunique() > 0:
        return jsonify({"status": "error", "message": f"Column '{column_name}' appears to be numerical. Cannot apply One-Hot Encoding."}), 400
    
    # Ensure it's a type that can be one-hot encoded
    if not (np.issubdtype(current_dataframe[column_name].dtype, object) or pd.api.types.is_categorical_dtype(current_dataframe[column_name])):
        return jsonify({"status": "error", "message": f"Column '{column_name}' is not suitable for One-Hot Encoding (must be object/category type). It is currently {current_dataframe[column_name].dtype}."}), 400

    try:
        # OneHotEncoder handles NaNs by default if drop='first' or handle_unknown='ignore'
        # It creates columns for valid categories and leaves NaNs as all zeros in the encoded columns
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        # We need to reshape the column for fit_transform
        encoded_data = encoder.fit_transform(current_dataframe[[column_name]])
        
        # Get feature names from the encoder
        feature_names = encoder.get_feature_names_out([column_name])
        
        encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=current_dataframe.index)

        # Drop the original column and concatenate the new one-hot encoded columns
        current_dataframe = pd.concat([current_dataframe.drop(columns=[column_name]), encoded_df], axis=1)
        
        message = f"One-hot encoding applied to '{column_name}'. New columns created."
        profiling_results = perform_profiling_and_plotting(current_dataframe)
        return jsonify({"status": "success", "message": message, **profiling_results}), 200
    except Exception as e:
        print(f"Error applying one-hot encoding: {e}")
        return jsonify({"status": "error", "message": f"Error applying one-hot encoding to '{column_name}': {str(e)}"}), 500

@app.route('/transform/frequency_encode', methods=['POST'])
def frequency_encode():
    """
    Applies frequency encoding to a specified categorical column.
    """
    global current_dataframe

    if current_dataframe is None:
        return jsonify({"status": "error", "message": "No data uploaded. Please upload a CSV first."}), 400

    data = request.json
    column_name = data.get('column_name')

    if not column_name:
        return jsonify({"status": "error", "message": "Missing 'column_name' parameter."}), 400

    if column_name not in current_dataframe.columns:
        return jsonify({"status": "error", "message": f"Column '{column_name}' not found."}), 400
    
    # Check if the column is truly categorical or object type, not numerical
    if pd.to_numeric(current_dataframe[column_name], errors='coerce').notna().all() and current_dataframe[column_name].nunique() > 0:
        return jsonify({"status": "error", "message": f"Column '{column_name}' appears to be numerical. Cannot apply Frequency Encoding."}), 400
    
    if not (np.issubdtype(current_dataframe[column_name].dtype, object) or pd.api.types.is_categorical_dtype(current_dataframe[column_name])):
        return jsonify({"status": "error", "message": f"Column '{column_name}' is not suitable for Frequency Encoding (must be object/category type). It is currently {current_dataframe[column_name].dtype}."}), 400

    try:
        # Calculate frequency mapping, handling NaNs if they exist in the column
        freq_map = current_dataframe[column_name].value_counts(dropna=False).to_dict()
        new_column_name = f"{column_name}_freq_encoded"
        current_dataframe[new_column_name] = current_dataframe[column_name].map(freq_map)

        message = f"Frequency encoding applied to '{column_name}'. New column '{new_column_name}' created."
        profiling_results = perform_profiling_and_plotting(current_dataframe)
        return jsonify({"status": "success", "message": message, **profiling_results}), 200
    except Exception as e:
        print(f"Error applying frequency encoding: {e}")
        return jsonify({"status": "error", "message": f"Error applying frequency encoding to '{column_name}': {str(e)}"}), 500

@app.route('/transform/handle_outliers', methods=['POST'])
def handle_outliers():
    """
    Handles outliers in a specified numerical column using winsorization or trimming.
    """
    global current_dataframe

    if current_dataframe is None:
        return jsonify({"status": "error", "message": "No data uploaded. Please upload a CSV first."}), 400

    data = request.json
    column_name = data.get('column_name')
    method = data.get('method') # 'winsorize', 'trim'
    lower_bound_quantile = data.get('lower_bound_quantile', 0.01) # Default 1st percentile
    upper_bound_quantile = data.get('upper_bound_quantile', 0.99) # Default 99th percentile

    if not column_name or not method:
        return jsonify({"status": "error", "message": "Missing 'column_name' or 'method' parameter."}), 400

    if column_name not in current_dataframe.columns:
        return jsonify({"status": "error", "message": f"Column '{column_name}' not found."}), 400

    if not np.issubdtype(current_dataframe[column_name].dtype, np.number):
        return jsonify({"status": "error", "message": f"Column '{column_name}' is not numerical. Cannot apply outlier handling."}), 400

    try:
        series = current_dataframe[column_name].dropna()
        if series.empty:
            return jsonify({"status": "success", "message": f"Column '{column_name}' has no non-missing values for outlier handling.", **perform_profiling_and_plotting(current_dataframe)}), 200

        lower_bound = series.quantile(lower_bound_quantile)
        upper_bound = series.quantile(upper_bound_quantile)

        if method == 'winsorize':
            # Cap values at the specified quantiles
            current_dataframe[column_name] = current_dataframe[column_name].clip(lower=lower_bound, upper=upper_bound)
            message = f"Outliers in '{column_name}' winsorized between {lower_bound_quantile*100}% and {upper_bound_quantile*100}% quantiles."
        elif method == 'trim':
            initial_rows = current_dataframe.shape[0]
            # Create a boolean mask for rows to keep
            rows_to_keep = (current_dataframe[column_name].isnull()) | \
                           ((current_dataframe[column_name] >= lower_bound) & (current_dataframe[column_name] <= upper_bound))
            current_dataframe = current_dataframe[rows_to_keep].copy() # Use .copy() to avoid SettingWithCopyWarning
            rows_removed = initial_rows - current_dataframe.shape[0]
            message = f"Outliers in '{column_name}' trimmed. {rows_removed} rows removed."
        else:
            return jsonify({"status": "error", "message": f"Invalid outlier handling method: {method}."}), 400
        
        profiling_results = perform_profiling_and_plotting(current_dataframe)
        return jsonify({"status": "success", "message": message, **profiling_results}), 200
    except Exception as e:
        print(f"Error handling outliers: {e}")
        return jsonify({"status": "error", "message": f"Error handling outliers in '{column_name}': {str(e)}"}), 500


@app.route('/feature_engineer/create_interaction', methods=['POST'])
def create_interaction_feature():
    """
    Creates an interaction feature by multiplying two specified numerical columns.
    """
    global current_dataframe

    if current_dataframe is None:
        return jsonify({"status": "error", "message": "No data uploaded. Please upload a CSV first."}), 400

    data = request.json
    col1 = data.get('column1')
    col2 = data.get('column2')

    if not col1 or not col2:
        return jsonify({"status": "error", "message": "Missing 'column1' or 'column2' parameter."}), 400

    if col1 not in current_dataframe.columns or col2 not in current_dataframe.columns:
        return jsonify({"status": "error", "message": "One or both columns not found."}), 400

    if not (np.issubdtype(current_dataframe[col1].dtype, np.number) and np.issubdtype(current_dataframe[col2].dtype, np.number)):
        return jsonify({"status": "error", "message": "Both columns must be numerical for interaction feature."}), 400
    
    if col1 == col2:
         return jsonify({"status": "error", "message": "Cannot create interaction feature with the same column."}), 400

    try:
        new_feature_name = f"{col1}_x_{col2}"
        current_dataframe[new_feature_name] = current_dataframe[col1] * current_dataframe[col2]
        message = f"Interaction feature '{new_feature_name}' created successfully."
        profiling_results = perform_profiling_and_plotting(current_dataframe)
        return jsonify({"status": "success", "message": message, **profiling_results}), 200
    except Exception as e:
        print(f"Error creating interaction feature: {e}")
        return jsonify({"status": "error", "message": f"Error creating interaction feature: {str(e)}"}), 500

@app.route('/feature_engineer/create_ratio', methods=['POST'])
def create_ratio_feature():
    """
    Creates a ratio feature by dividing two specified numerical columns.
    """
    global current_dataframe

    if current_dataframe is None:
        return jsonify({"status": "error", "message": "No data uploaded. Please upload a CSV first."}), 400

    data = request.json
    numerator_col = data.get('numerator_column')
    denominator_col = data.get('denominator_column')

    if not numerator_col or not denominator_col:
        return jsonify({"status": "error", "message": "Missing numerator or denominator column."}), 400

    if numerator_col not in current_dataframe.columns or denominator_col not in current_dataframe.columns:
        return jsonify({"status": "error", "message": "One or both columns not found."}), 400

    if not (np.issubdtype(current_dataframe[numerator_col].dtype, np.number) and np.issubdtype(current_dataframe[denominator_col].dtype, np.number)):
        return jsonify({"status": "error", "message": "Both columns must be numerical for ratio feature."}), 400
    
    if numerator_col == denominator_col:
        return jsonify({"status": "error", "message": "Cannot create ratio feature with the same column."}), 400

    try:
        # Handle division by zero by replacing infinities with NaN
        new_feature_name = f"{numerator_col}_div_{denominator_col}"
        current_dataframe[new_feature_name] = current_dataframe[numerator_col] / current_dataframe[denominator_col]
        current_dataframe[new_feature_name].replace([np.inf, -np.inf], np.nan, inplace=True) # Replace inf with NaN

        message = f"Ratio feature '{new_feature_name}' created successfully."
        profiling_results = perform_profiling_and_plotting(current_dataframe)
        return jsonify({"status": "success", "message": message, **profiling_results}), 200
    except Exception as e:
        print(f"Error creating ratio feature: {e}")
        return jsonify({"status": "error", "message": f"Error creating ratio feature: {str(e)}"}), 500

@app.route('/feature_engineer/create_lagged_feature', methods=['POST'])
def create_lagged_feature():
    """
    Creates a lagged feature for a specified column, grouped by country (or other ID) and ordered by year.
    Requires 'Country name' and 'Year' columns to be present.
    """
    global current_dataframe

    if current_dataframe is None:
        return jsonify({"status": "error", "message": "No data uploaded. Please upload a CSV first."}), 400

    data = request.json
    column_name = data.get('column_name')
    group_col = data.get('group_column')
    time_col = data.get('time_column')
    periods = int(data.get('periods', 1)) # Default lag by 1 period

    if not column_name or not group_col or not time_col:
        return jsonify({"status": "error", "message": "Missing column_name, group_column, or time_column."}), 400

    if not all(col in current_dataframe.columns for col in [column_name, group_col, time_col]):
        return jsonify({"status": "error", "message": "One or more specified columns (feature, group, or time) not found."}), 400
    
    if not np.issubdtype(current_dataframe[column_name].dtype, np.number):
        return jsonify({"status": "error", "message": f"Lagged feature can only be created for numerical columns. '{column_name}' is not numerical."}), 400

    try:
        # Ensure time_col is numeric or convertible to datetime for sorting
        if not np.issubdtype(current_dataframe[time_col].dtype, np.number):
            try:
                current_dataframe[time_col] = pd.to_datetime(current_dataframe[time_col])
            except Exception:
                return jsonify({"status": "error", "message": f"Time column '{time_col}' must be numerical or convertible to datetime."}), 400

        # Sort by group and time for correct lagging
        # Using .copy() to avoid SettingWithCopyWarning
        df_sorted = current_dataframe.sort_values(by=[group_col, time_col]).copy()
        
        new_feature_name = f"{column_name}_lag_{periods}"
        df_sorted[new_feature_name] = df_sorted.groupby(group_col)[column_name].shift(periods=periods)
        
        # Merge back to original index to maintain dataframe order
        # We need to drop the newly created lagged column from current_dataframe first, if it exists
        if new_feature_name in current_dataframe.columns:
            current_dataframe.drop(columns=[new_feature_name], inplace=True)

        current_dataframe = current_dataframe.merge(df_sorted[[group_col, time_col, new_feature_name]], 
                                                    on=[group_col, time_col], 
                                                    how='left',
                                                    suffixes=('', '_y')) # Add suffix to avoid conflicts if columns have same name


        # Drop any duplicated columns that might arise from merge (e.g., if new_feature_name existed before)
        current_dataframe = current_dataframe.loc[:,~current_dataframe.columns.duplicated()].copy()


        message = f"Lagged feature '{new_feature_name}' created successfully for '{column_name}'."
        profiling_results = perform_profiling_and_plotting(current_dataframe)
        return jsonify({"status": "success", "message": message, **profiling_results}), 200
    except Exception as e:
        print(f"Error creating lagged feature: {e}")
        return jsonify({"status": "error", "message": f"Error creating lagged feature: {str(e)}"}), 500


@app.route('/get_ai_insights', methods=['POST'])
def get_ai_insights():
    """
    Generates AI-powered insights, suggestions, and model recommendations using Gemini.
    """
    global current_dataframe

    if current_dataframe is None:
        return jsonify({"status": "error", "message": "No data uploaded. Please upload a CSV first."}), 400

    try:
        profiling_data = perform_profiling_and_plotting(current_dataframe)

        prompt_parts = []
        prompt_parts.append("As an expert data scientist and machine learning engineer, analyze the following dataset profile. Provide your insights in a clear, well-structured, and easy-to-read format, using Markdown for headings, bolding, and paragraphs to enhance readability. Avoid excessive bullet points for high-level summaries.\n\n")

        prompt_parts.append("### Dataset Overview\n")
        prompt_parts.append(f"This dataset contains {profiling_data['data_summary']['num_rows']} rows and {profiling_data['data_summary']['num_cols']} columns.\n\n")

        prompt_parts.append("#### Column Data Types and Missing Values\n")
        for col_name in profiling_data['data_summary']['column_names']:
            dtype = profiling_data['data_summary']['data_types'][col_name]
            missing_count = profiling_data['data_summary']['missing_values'][col_name]
            missing_percent = profiling_data['data_summary']['missing_percentages'][col_name]
            prompt_parts.append(f"- **{col_name}**: Type='{dtype}', Missing={missing_count} ({missing_percent}%)\n")
        prompt_parts.append("\n")

        prompt_parts.append("#### Numerical Feature Descriptive Statistics\n")
        numerical_cols = [col for col, dtype in profiling_data['data_summary']['data_types'].items() if 'int' in dtype or 'float' in dtype]
        if numerical_cols:
            metrics_order = ['count', 'mean', 'std', 'min', '1%', '5%', '25%', '50%', '75%', '95%', '99%', 'max', 'skew', 'kurtosis']
            prompt_parts.append("```\n") # Markdown code block for table
            # Table Header
            header_row = ["Metric"] + numerical_cols
            prompt_parts.append(" | ".join(header_row) + "\n")
            prompt_parts.append("---" * (len(header_row)) + "\n")
            # Table Rows
            for metric in metrics_order:
                row_values = [metric.capitalize().replace('%', ' Percentile')]
                for col in numerical_cols:
                    val = profiling_data['descriptive_statistics'].get(col, {}).get(metric)
                    if val is not None:
                        if isinstance(val, (int, float)):
                            val_str = f"{val:.2f}" if isinstance(val, float) else str(int(val))
                        else:
                            val_str = str(val)
                        row_values.append(val_str)
                    else:
                        row_values.append("N/A")
                prompt_parts.append(" | ".join(row_values) + "\n")
            prompt_parts.append("```\n\n")
        else:
            prompt_parts.append("No numerical columns found to generate descriptive statistics.\n\n")

        prompt_parts.append("#### Numerical Feature Skewness (Specific Values)\n")
        if profiling_data['skewness_info'] and any(v is not None for v in profiling_data['skewness_info'].values()):
            prompt_parts.append("```\n")
            skew_cols = list(profiling_data['skewness_info'].keys())
            prompt_parts.append(" | ".join(["Column"] + skew_cols) + "\n")
            prompt_parts.append("---" * (len(skew_cols) + 1) + "\n")
            skew_values = ["Skewness"]
            for col in skew_cols:
                val = profiling_data['skewness_info'].get(col)
                skew_values.append(f"{val:.2f}" if val is not None else "N/A")
            prompt_parts.append(" | ".join(skew_values) + "\n")
            prompt_parts.append("```\n\n")
        else:
            prompt_parts.append("No numerical columns or no skewness information available.\n\n")


        prompt_parts.append("#### Numerical Feature Outlier Information (IQR Method)\n")
        if profiling_data['outlier_info'] and any(info and info['count'] > 0 for info in profiling_data['outlier_info'].values()):
            prompt_parts.append("```\n")
            outlier_cols = [col for col, info in profiling_data['outlier_info'].items() if info and info['count'] > 0]
            prompt_parts.append(" | ".join(["Column", "Count", "Percentage", "Lower Bound", "Upper Bound"]) + "\n")
            prompt_parts.append("---" * 5 + "\n")
            for col in outlier_cols:
                info = profiling_data['outlier_info'][col]
                prompt_parts.append(f"{col} | {info['count']} | {info['percentage']}% | {info['lower_bound']} | {info['upper_bound']}\n")
            prompt_parts.append("```\n\n")
        else:
            prompt_parts.append("No significant outliers detected in numerical columns based on the IQR method.\n\n")


        prompt_parts.append("#### Categorical Feature Value Counts (Top 10)\n")
        if profiling_data['column_details']:
            for col, details in profiling_data['column_details'].items():
                prompt_parts.append(f"**{col}** (Unique: {details['unique_count']})\n")
                if details['has_more_values']:
                    prompt_parts.append(f"*(Note: More than {details['unique_count']} unique values exist, showing top 10)*\n")
                prompt_parts.append("```\n")
                prompt_parts.append(" | ".join(["Value", "Count", "Percentage"]) + "\n")
                prompt_parts.append("---" * 3 + "\n")
                for val, count in details['value_counts'].items():
                    percent = details['value_percentages'].get(val, 0)
                    prompt_parts.append(f"'{val}' | {count} | {percent}%\n")
                prompt_parts.append("```\n\n")
            prompt_parts.append("\n")
        else:
            prompt_parts.append("No categorical columns found.\n\n")

        prompt_parts.append("#### Numerical Feature Correlation Matrix\n")
        if profiling_data['correlation_matrix_data']:
            corr_cols = list(profiling_data['correlation_matrix_data'][0].keys()) if profiling_data['correlation_matrix_data'] else []
            if corr_cols:
                prompt_parts.append("```\n")
                prompt_parts.append(" | ".join(["Feature"] + corr_cols) + "\n")
                prompt_parts.append("---" * (len(corr_cols) + 1) + "\n")
                for i, row_dict in enumerate(profiling_data['correlation_matrix_data']):
                    row_name = corr_cols[i]
                    row_values = [row_name]
                    for col in corr_cols:
                        val = row_dict.get(col)
                        if isinstance(val, (int, float)):
                            row_values.append(f"{val:.2f}")
                        else:
                            row_values.append("N/A")
                    prompt_parts.append(" | ".join(row_values) + "\n")
                prompt_parts.append("```\n\n")
            else:
                prompt_parts.append("Numerical correlation matrix data is empty.\n\n")
        else:
            prompt_parts.append("No numerical correlation matrix (less than 2 numerical columns to compute correlation).\n\n")


        # --- Detailed Instructions for Gemini's Response ---
        prompt_parts.append("--- Data Scientist's Report ---\n\n")
        prompt_parts.append("Based on the comprehensive data profile provided, please generate a **step-by-step data preprocessing and feature engineering procedure**. This procedure should be designed for a non-data scientist to follow easily using the available functions in the web application. Focus on the most appropriate and common steps to prepare this dataset for a typical machine learning task (e.g., prediction or analysis).\n\n")

        prompt_parts.append("### Step-by-Step Preprocessing Procedure\n")
        prompt_parts.append("Provide a numbered list of actions. For each step, clearly state:\n")
        prompt_parts.append("1.  **The specific action to take.**\n")
        prompt_parts.append("2.  **Which column(s) it applies to.**\n")
        prompt_parts.append("3.  **The recommended method/strategy/type** (e.g., 'mean' for imputation, 'standard' for scaling, 'one-hot encode' for encoding).\n")
        prompt_parts.append("4.  **A brief justification** for why this step is recommended for this specific dataset based on the profile data.\n")
        prompt_parts.append("Prioritize steps logically (e.g., handle missing values before scaling). If a step is not necessary, explicitly state that (e.g., 'No missing values found, skipping imputation').\n\n")
        
        prompt_parts.append("### General Recommendations & Considerations\n")
        prompt_parts.append("After the step-by-step guide, provide additional high-level advice on:\n")
        prompt_parts.append("- **Potential Target Variables**: Suggest 1-2 columns that could likely serve as a target variable for predictive modeling, with a brief reason.\n")
        prompt_parts.append("- **Suitable Model Types**: Based on the data characteristics, suggest whether regression, classification, or clustering models would generally be appropriate.\n")
        prompt_parts.append("- **Important Reminders**: Briefly reiterate crucial data science principles like preventing data leakage (especially if lagged features or target encoding were suggested) and the importance of data splitting.\n\n")

        prompt_parts.append("Ensure your entire response is formatted for easy reading and professional presentation. Use proper Markdown for structure.")

        gemini_prompt = "".join(prompt_parts)
        print("--- Gemini Prompt ---")
        print(gemini_prompt)
        print("---------------------")

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(gemini_prompt)

        ai_response = response.text
        return jsonify({"status": "success", "message": "AI insights generated!", "ai_insights": ai_response}), 200

    except Exception as e:
        print(f"Error generating AI insights: {e}")
        return jsonify({"status": "error", "message": f"Error generating AI insights: {str(e)}"}), 500

@app.route('/download_csv', methods=['GET'])
def download_csv():
    """
    Allows the user to download the current state of the processed DataFrame as a CSV.
    """
    global current_dataframe

    if current_dataframe is None:
        return jsonify({"status": "error", "message": "No data available to download."}), 400

    try:
        # Create a BytesIO object to save the CSV data in-memory
        output = io.StringIO()
        current_dataframe.to_csv(output, index=False)
        output.seek(0) # Rewind to the beginning of the stream

        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='processed_data.csv'
        ), 200
    except Exception as e:
        print(f"Error downloading CSV: {e}")
        return jsonify({"status": "error", "message": f"Error downloading processed CSV: {str(e)}"}), 500

@app.route('/export_code', methods=['POST'])
def export_code():
    """
    Generates Python code based on the sequence of transformations applied.
    """
    data = request.json
    transformations = data.get('transformations', [])

    if not transformations:
        return jsonify({"status": "error", "message": "No transformations to export."}), 400

    code_lines = [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer",
        "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, PowerTransformer",
        "from scipy.stats import boxcox",
        "",
        "# --- Load your data (replace 'your_data.csv' with your actual file path) ---",
        "try:",
        "    df = pd.read_csv('your_data.csv')",
        "except FileNotFoundError:",
        "    print(\"Error: 'your_data.csv' not found. Please ensure the CSV file is in the same directory or provide the full path.\")",
        "    exit()",
        "",
        "# --- Applied Data Transformations ---"
    ]

    for i, action in enumerate(transformations):
        endpoint = action['endpoint']
        payload = action['payload']
        code_lines.append(f"\n# Transformation {i+1}: {endpoint.replace('/',' ').strip().replace('_', ' ').title()}")
        
        if endpoint == '/preprocess/remove_duplicates':
            code_lines.append("df.drop_duplicates(inplace=True)")
            code_lines.append("print(f\"Removed duplicate rows. New shape: {df.shape}\")")
        
        elif endpoint == '/preprocess/impute_missing':
            col = payload.get('column')
            strategy = payload.get('strategy')
            knn_neighbors = payload.get('k_neighbors', 5)
            
            if col == 'all_numerical':
                code_lines.append("numerical_cols = df.select_dtypes(include=np.number).columns.tolist()")
                target_cols_str = "numerical_cols"
            elif col == 'all_categorical':
                code_lines.append("categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()")
                target_cols_str = "categorical_cols"
            else:
                target_cols_str = f"['{col}']"

            if strategy == 'mean':
                code_lines.append(f"for col in {target_cols_str}:")
                code_lines.append(f"    if np.issubdtype(df[col].dtype, np.number):")
                code_lines.append(f"        df[col].fillna(df[col].mean(), inplace=True)")
                code_lines.append(f"print(f\"Imputed missing values in {target_cols_str} with mean.\")")
            elif strategy == 'median':
                code_lines.append(f"for col in {target_cols_str}:")
                code_lines.append(f"    if np.issubdtype(df[col].dtype, np.number):")
                code_lines.append(f"        df[col].fillna(df[col].median(), inplace=True)")
                code_lines.append(f"print(f\"Imputed missing values in {target_cols_str} with median.\")")
            elif strategy == 'mode':
                code_lines.append(f"for col in {target_cols_str}:")
                code_lines.append(f"    mode_val = df[col].mode()[0]")
                code_lines.append(f"    df[col].fillna(mode_val, inplace=True)")
                code_lines.append(f"print(f\"Imputed missing values in {target_cols_str} with mode.\")")
            elif strategy == 'knn':
                code_lines.append(f"imputation_cols = [c for c in {target_cols_str} if np.issubdtype(df[c].dtype, np.number)]")
                code_lines.append(f"if imputation_cols:")
                code_lines.append(f"    knn_imputer = KNNImputer(n_neighbors={knn_neighbors})")
                code_lines.append(f"    df[imputation_cols] = knn_imputer.fit_transform(df[imputation_cols])")
                code_lines.append(f"    print(f\"Imputed missing values in {{imputation_cols}} using KNN (k={knn_neighbors}).\")")
            elif strategy == 'mice':
                code_lines.append(f"imputation_cols = [c for c in {target_cols_str} if np.issubdtype(df[c].dtype, np.number)]")
                code_lines.append(f"if imputation_cols:")
                code_lines.append(f"    mice_imputer = IterativeImputer(max_iter=10, random_state=0)")
                code_lines.append(f"    df[imputation_cols] = mice_imputer.fit_transform(df[imputation_cols])")
                code_lines.append(f"    print(f\"Imputed missing values in {{imputation_cols}} using MICE.\")")
        
        elif endpoint == '/preprocess/remove_missing_rows':
            code_lines.append("initial_rows = df.shape[0]")
            code_lines.append("df.dropna(inplace=True)")
            code_lines.append("rows_removed = initial_rows - df.shape[0]")
            code_lines.append("print(f\"Removed rows with missing values. {{rows_removed}} rows deleted.\")")
        
        elif endpoint == '/preprocess/remove_missing_cols':
            code_lines.append("cols_before = set(df.columns)")
            code_lines.append("df.dropna(axis=1, inplace=True)")
            code_lines.append("cols_after = set(df.columns)")
            code_lines.append("cols_removed = list(cols_before - cols_after)")
            code_lines.append("print(f\"Removed columns with missing values. Columns deleted: {{', '.join(cols_removed) if cols_removed else 'None'}}.\")")

        elif endpoint == '/data/drop_column':
            col_name = payload.get('column_name')
            code_lines.append(f"if '{col_name}' in df.columns:")
            code_lines.append(f"    df.drop(columns=['{col_name}'], inplace=True)")
            code_lines.append(f"    print(f\"Column '{{'{col_name}'}}' dropped successfully.\")")
            code_lines.append(f"else:")
            code_lines.append(f"    print(f\"Column '{{'{col_name}'}}' not found for dropping.\")")
        
        elif endpoint == '/transform/scale_features':
            cols = payload.get('columns')
            scaler_type = payload.get('scaler_type')
            
            code_lines.append(f"columns_to_scale = {cols}")
            code_lines.append("if columns_to_scale:")
            if scaler_type == 'standard':
                code_lines.append("    scaler = StandardScaler()")
            elif scaler_type == 'minmax':
                code_lines.append("    scaler = MinMaxScaler()")
            elif scaler_type == 'robust':
                code_lines.append("    scaler = RobustScaler()")
            code_lines.append("    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])")
            code_lines.append(f"    print(f\"Features {{columns_to_scale}} scaled using {scaler_type} scaler.\")")
        
        elif endpoint == '/transform/apply_skew_transform':
            col_name = payload.get('column_name')
            transform_type = payload.get('transform_type')
            
            code_lines.append(f"if '{col_name}' in df.columns and np.issubdtype(df['{col_name}'].dtype, np.number):")
            code_lines.append(f"    col_series = df['{col_name}'].dropna()")
            if transform_type == 'log':
                code_lines.append(f"    transformed_values = np.log1p(col_series)")
                code_lines.append(f"    df['{col_name}'] = transformed_values")
                code_lines.append(f"    print(f\"Log transformation applied to '{{'{col_name}'}}'.\")")
            elif transform_type == 'sqrt':
                code_lines.append(f"    transformed_values = np.sqrt(col_series)")
                code_lines.append(f"    df['{col_name}'] = transformed_values")
                code_lines.append(f"    print(f\"Square root transformation applied to '{{'{col_name}'}}'.\")")
            elif transform_type == 'boxcox':
                code_lines.append(f"    if (col_series <= 0).any():")
                code_lines.append(f"        print(f\"Warning: Box-Cox transformation requires positive values. Skipping for '{{'{col_name}'}}'.\")")
                code_lines.append(f"    else:")
                code_lines.append(f"        transformed_values, _ = boxcox(col_series)")
                code_lines.append(f"        df['{col_name}'] = transformed_values")
                code_lines.append(f"        print(f\"Box-Cox transformation applied to '{{'{col_name}'}}'.\")")
            elif transform_type == 'yeo-johnson':
                code_lines.append(f"    pt = PowerTransformer(method='yeo-johnson')")
                code_lines.append(f"    transformed_values = pt.fit_transform(col_series.values.reshape(-1, 1)).flatten()")
                code_lines.append(f"    df['{col_name}'] = transformed_values")
                code_lines.append(f"    print(f\"Yeo-Johnson transformation applied to '{{'{col_name}'}}'.\")")
            code_lines.append(f"else:")
            code_lines.append(f"    print(f\"Column '{{'{col_name}'}}' not numerical or not found for skew transformation.\")")

        elif endpoint == '/transform/one_hot_encode':
            col_name = payload.get('column_name')
            code_lines.append(f"if '{col_name}' in df.columns:")
            code_lines.append(f"    if pd.to_numeric(df['{col_name}'], errors='coerce').notna().all():")
            code_lines.append(f"        print(f\"Warning: Column '{{'{col_name}'}}' appears numerical. Skipping One-Hot Encoding.\")")
            code_lines.append(f"    else:")
            code_lines.append(f"        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')")
            code_lines.append(f"        encoded_data = encoder.fit_transform(df[['{col_name}']])")
            code_lines.append(f"        feature_names = encoder.get_feature_names_out(['{col_name}'])")
            code_lines.append(f"        encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)")
            code_lines.append(f"        df = pd.concat([df.drop(columns=['{col_name}']), encoded_df], axis=1)")
            code_lines.append(f"        print(f\"One-hot encoding applied to '{{'{col_name}'}}'.\")")
            code_lines.append(f"else:")
            code_lines.append(f"    print(f\"Column '{{'{col_name}'}}' not found for one-hot encoding.\")")

        elif endpoint == '/transform/frequency_encode':
            col_name = payload.get('column_name')
            code_lines.append(f"if '{col_name}' in df.columns:")
            code_lines.append(f"    if pd.to_numeric(df['{col_name}'], errors='coerce').notna().all():")
            code_lines.append(f"        print(f\"Warning: Column '{{'{col_name}'}}' appears numerical. Skipping Frequency Encoding.\")")
            code_lines.append(f"    else:")
            code_lines.append(f"        freq_map = df['{col_name}'].value_counts(dropna=False).to_dict()")
            code_lines.append(f"        new_column_name = f\"{{'{col_name}'}}_freq_encoded\"")
            code_lines.append(f"        df[new_column_name] = df['{col_name}'].map(freq_map)")
            code_lines.append(f"        print(f\"Frequency encoding applied to '{{'{col_name}'}}'. New column '{{new_column_name}}' created.\")")
            code_lines.append(f"else:")
            code_lines.append(f"    print(f\"Column '{{'{col_name}'}}' not found for frequency encoding.\")")
        
        elif endpoint == '/transform/handle_outliers':
            col_name = payload.get('column_name')
            method = payload.get('method')
            lower_quantile = payload.get('lower_bound_quantile')
            upper_quantile = payload.get('upper_bound_quantile')

            code_lines.append(f"if '{col_name}' in df.columns and np.issubdtype(df['{col_name}'].dtype, np.number):")
            code_lines.append(f"    series = df['{col_name}'].dropna()")
            code_lines.append(f"    if not series.empty:")
            code_lines.append(f"        lower_bound = series.quantile({lower_quantile})")
            code_lines.append(f"        upper_bound = series.quantile({upper_quantile})")
            if method == 'winsorize':
                code_lines.append(f"        df['{col_name}'] = df['{col_name}'].clip(lower=lower_bound, upper=upper_bound)")
                code_lines.append(f"        print(f\"Outliers in '{{'{col_name}'}}' winsorized between {lower_quantile*100}% and {upper_quantile*100}% quantiles.\")")
            elif method == 'trim':
                code_lines.append(f"        initial_rows = df.shape[0]")
                code_lines.append(f"        rows_to_keep = (df['{col_name}'].isnull()) | \\")
                code_lines.append(f"                       ((df['{col_name}'] >= lower_bound) & (df['{col_name}'] <= upper_bound))")
                code_lines.append(f"        df = df[rows_to_keep].copy()")
                code_lines.append(f"        rows_removed = initial_rows - df.shape[0]")
                code_lines.append(f"        print(f\"Outliers in '{{'{col_name}'}}' trimmed. {{rows_removed}} rows removed.\")")
            code_lines.append(f"    else:")
            code_lines.append(f"        print(f\"Column '{{'{col_name}'}}' has no non-missing values for outlier handling.\")")
            code_lines.append(f"else:")
            code_lines.append(f"    print(f\"Column '{{'{col_name}'}}' not numerical or not found for outlier handling.\")")

        elif endpoint == '/feature_engineer/create_interaction':
            col1 = payload.get('column1')
            col2 = payload.get('column2')
            new_feature_name = f"{col1}_x_{col2}"
            code_lines.append(f"if all(col in df.columns for col in ['{col1}', '{col2}']) and '{col1}' != '{col2}':")
            code_lines.append(f"    if np.issubdtype(df['{col1}'].dtype, np.number) and np.issubdtype(df['{col2}'].dtype, np.number):")
            code_lines.append(f"        df['{new_feature_name}'] = df['{col1}'] * df['{col2}']")
            code_lines.append(f"        print(f\"Interaction feature '{{'{new_feature_name}'}}' created successfully.\")")
            code_lines.append(f"    else:")
            code_lines.append(f"        print(f\"Columns '{{'{col1}'}}' and '{{'{col2}'}}' must be numerical for interaction feature.\")")
            code_lines.append(f"else:")
            code_lines.append(f"    print(f\"One or both columns ('{{'{col1}'}}', '{{'{col2}'}}') not found or are the same for interaction feature.\")")

        elif endpoint == '/feature_engineer/create_ratio':
            num_col = payload.get('numerator_column')
            den_col = payload.get('denominator_column')
            new_feature_name = f"{num_col}_div_{den_col}"
            code_lines.append(f"if all(col in df.columns for col in ['{num_col}', '{den_col}']) and '{num_col}' != '{den_col}':")
            code_lines.append(f"    if np.issubdtype(df['{num_col}'].dtype, np.number) and np.issubdtype(df['{den_col}'].dtype, np.number):")
            code_lines.append(f"        df['{new_feature_name}'] = df['{num_col}'] / df['{den_col}']")
            code_lines.append(f"        df['{new_feature_name}'].replace([np.inf, -np.inf], np.nan, inplace=True)")
            code_lines.append(f"        print(f\"Ratio feature '{{'{new_feature_name}'}}' created successfully.\")")
            code_lines.append(f"    else:")
            code_lines.append(f"        print(f\"Columns '{{'{num_col}'}}' and '{{'{den_col}'}}' must be numerical for ratio feature.\")")
            code_lines.append(f"else:")
            code_lines.append(f"    print(f\"One or both columns ('{{'{num_col}'}}', '{{'{den_col}'}}') not found or are the same for ratio feature.\")")

        elif endpoint == '/feature_engineer/create_lagged_feature':
            col_name = payload.get('column_name')
            group_col = payload.get('group_column')
            time_col = payload.get('time_column')
            periods = payload.get('periods')
            new_feature_name = f"{col_name}_lag_{periods}"

            code_lines.append(f"if all(col in df.columns for col in ['{col_name}', '{group_col}', '{time_col}']):")
            code_lines.append(f"    if np.issubdtype(df['{col_name}'].dtype, np.number):")
            code_lines.append(f"        # Ensure time_col is sortable (numeric or datetime)")
            code_lines.append(f"        if not np.issubdtype(df['{time_col}'].dtype, np.number):")
            code_lines.append(f"            try:")
            code_lines.append(f"                df['{time_col}'] = pd.to_datetime(df['{time_col}'])")
            code_lines.append(f"            except Exception:")
            code_lines.append(f"                print(f\"Warning: Time column '{{'{time_col}'}}' must be numerical or convertible to datetime. Skipping lagged feature.\")")
            code_lines.append(f"        ")
            code_lines.append(f"        # Sort by group and time for correct lagging")
            code_lines.append(f"        df_sorted_for_lag = df.sort_values(by=['{group_col}', '{time_col}']).copy()")
            code_lines.append(f"        df_sorted_for_lag['{new_feature_name}'] = df_sorted_for_lag.groupby('{group_col}')['{col_name}'].shift(periods={periods})")
            code_lines.append(f"        ")
            code_lines.append(f"        # Merge back to original index to maintain dataframe order and handle new column")
            code_lines.append(f"        if '{new_feature_name}' in df.columns:")
            code_lines.append(f"            df.drop(columns=['{new_feature_name}'], inplace=True)")
            code_lines.append(f"        df = df.merge(df_sorted_for_lag[['{group_col}', '{time_col}', '{new_feature_name}']], ")
            code_lines.append(f"                        on=['{group_col}', '{time_col}'], how='left', suffixes=('', '_y'))")
            code_lines.append(f"        df = df.loc[:,~df.columns.duplicated()].copy()") # Handle potential duplicated columns from merge
            code_lines.append(f"        print(f\"Lagged feature '{{'{new_feature_name}'}}' created successfully for '{{'{col_name}'}}'.\")")
            code_lines.append(f"    else:")
            code_lines.append(f"        print(f\"Lagged feature can only be created for numerical columns. '{{'{col_name}'}}' is not numerical.\")")
            code_lines.append(f"else:")
            code_lines.append(f"    print(f\"One or more specified columns (feature: '{{'{col_name}'}}', group: '{{'{group_col}'}}', or time: '{{'{time_col}'}}') not found for lagged feature.\")")

    code_lines.append("\n# --- Final Processed DataFrame Head ---")
    code_lines.append("print(\"\\nFinal Processed DataFrame Head:\")")
    code_lines.append("print(df.head())")
    
    code_lines.append("\n# --- Save Processed Data (Optional) ---")
    code_lines.append("# df.to_csv('final_processed_data.csv', index=False)")
    code_lines.append("# print(\"\\nProcessed data saved to 'final_processed_data.csv'\")")

    return jsonify({"status": "success", "message": "Python code generated!", "code": "\n".join(code_lines)}), 200

@app.route('/train_model', methods=['POST'])
def train_model():
    """
    Trains a selected machine learning model and returns evaluation metrics and plots.
    """
    global current_dataframe

    if current_dataframe is None:
        return jsonify({"status": "error", "message": "No data uploaded. Please upload a CSV first."}), 400

    data = request.json
    target_variable = data.get('target_variable')
    test_size = data.get('test_size', 0.2)
    random_state = data.get('random_state', 42)
    selected_model = data.get('selected_model')

    if not target_variable or not selected_model:
        return jsonify({"status": "error", "message": "Target variable and selected model are required."}), 400

    if target_variable not in current_dataframe.columns:
        return jsonify({"status": "error", "message": f"Target variable '{target_variable}' not found in the data."}), 400

    try:
        # Separate features (X) and target (y)
        y = current_dataframe[target_variable].copy()
        X = current_dataframe.drop(columns=[target_variable]).copy()

        # IMPORTANT: Handle non-numeric columns in X
        # For simplicity, we will drop non-numeric columns from X for now.
        # In a more advanced scenario, these would need to be preprocessed (e.g., encoded).
        non_numeric_cols_in_X = X.select_dtypes(exclude=np.number).columns.tolist()
        if non_numeric_cols_in_X:
            print(f"Warning: Dropping non-numerical columns from features: {non_numeric_cols_in_X}. Consider encoding them if they are important.")
            X = X.drop(columns=non_numeric_cols_in_X)
        
        # Drop rows with any remaining NaNs in X or y for model training
        # This is a simple approach; proper imputation should be done earlier.
        # Align X and y after dropping NaNs to ensure consistency
        combined_df = pd.concat([X, y], axis=1).dropna()
        if combined_df.empty:
            return jsonify({"status": "error", "message": "After dropping non-numeric columns and rows with missing values, no data remains for training. Please check your preprocessing."}), 400
        
        X = combined_df.drop(columns=[target_variable])
        y = combined_df[target_variable]

        if X.empty:
            return jsonify({"status": "error", "message": "No features remaining after preprocessing for model training."}), 400
        
        metrics = {}
        plots = {} # Dictionary to store plot images

        if selected_model in ['logistic_regression', 'decision_tree_classifier']:
            # For classification, ensure target is numerical (e.g., 0/1)
            # Use LabelEncoder if the target is categorical string
            if not np.issubdtype(y.dtype, np.number):
                le = LabelEncoder()
                try:
                    y_encoded = le.fit_transform(y)
                    metrics['class_labels'] = le.classes_.tolist() # Store original class labels
                    y = pd.Series(y_encoded, index=y.index, name=y.name)
                except Exception as e:
                    return jsonify({"status": "error", "message": f"Could not encode target variable '{target_variable}' for classification: {str(e)}. Ensure it's suitable for encoding."}), 400

            if len(y.unique()) < 2:
                return jsonify({"status": "error", "message": f"Classification models require at least two unique values in the target variable, but '{target_variable}' has only {len(y.unique())} unique value(s). Please choose a different target or model."}), 400
            
            # For classification, we need to handle multi-class if target has > 2 unique values
            if len(y.unique()) > 2:
                metrics['is_multiclass'] = True
                average_type = 'weighted'
            else:
                metrics['is_multiclass'] = False
                average_type = 'binary' # Or 'macro' / 'micro' if preferred

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        model_instance = None
        
        if selected_model == 'linear_regression':
            model_instance = LinearRegression()
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)
            metrics['model_type'] = 'Regression'
            metrics['mse'] = round(mean_squared_error(y_test, y_pred), 4)
            metrics['r2_score'] = round(r2_score(y_test, y_pred), 4)
            metrics['message'] = f"Linear Regression model trained successfully for '{target_variable}'."

            # Residual Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=y_pred, y=(y_test - y_pred), ax=ax)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('Residual Plot')
            plots['residual_plot_image'] = generate_plot_image(fig)


        elif selected_model == 'logistic_regression':
            model_instance = LogisticRegression(random_state=random_state, max_iter=1000, solver='liblinear') # Increased max_iter for convergence
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)
            metrics['model_type'] = 'Classification'
            metrics['accuracy'] = round(accuracy_score(y_test, y_pred), 4)
            metrics['precision'] = round(precision_score(y_test, y_pred, average=average_type, zero_division=0), 4)
            metrics['recall'] = round(recall_score(y_test, y_pred, average=average_type, zero_division=0), 4)
            metrics['f1_score'] = round(f1_score(y_test, y_pred, average=average_type, zero_division=0), 4)
            metrics['message'] = f"Logistic Regression model trained successfully for '{target_variable}'."

            # Confusion Matrix Plot
            cm = confusion_matrix(y_test, y_pred, labels=model_instance.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=metrics.get('class_labels', model_instance.classes_))
            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(cmap=plt.cm.Blues, ax=ax)
            ax.set_title('Confusion Matrix')
            plots['confusion_matrix_image'] = generate_plot_image(fig)

            # ROC Curve (for binary classification only)
            if not metrics['is_multiclass']:
                y_prob = model_instance.predict_proba(X_test)[:, 1] # Probability of the positive class
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                metrics['roc_auc'] = round(roc_auc, 4)

                fig, ax = plt.subplots(figsize=(8, 6))
                RocCurveDisplay.from_predictions(y_test, y_prob, name=f"{selected_model} (AUC = {roc_auc:.2f})", ax=ax)
                ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.50)')
                ax.set_title('ROC Curve')
                plots['roc_curve_image'] = generate_plot_image(fig)


        elif selected_model == 'decision_tree_classifier':
            model_instance = DecisionTreeClassifier(random_state=random_state)
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)
            metrics['model_type'] = 'Classification'
            metrics['accuracy'] = round(accuracy_score(y_test, y_pred), 4)
            metrics['precision'] = round(precision_score(y_test, y_pred, average=average_type, zero_division=0), 4)
            metrics['recall'] = round(recall_score(y_test, y_pred, average=average_type, zero_division=0), 4)
            metrics['f1_score'] = round(f1_score(y_test, y_pred, average=average_type, zero_division=0), 4)
            metrics['message'] = f"Decision Tree Classifier trained successfully for '{target_variable}'."

            # Confusion Matrix Plot
            cm = confusion_matrix(y_test, y_pred, labels=model_instance.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=metrics.get('class_labels', model_instance.classes_))
            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(cmap=plt.cm.Blues, ax=ax)
            ax.set_title('Confusion Matrix')
            plots['confusion_matrix_image'] = generate_plot_image(fig)
            
            # ROC Curve (for binary classification only)
            if not metrics['is_multiclass'] and hasattr(model_instance, 'predict_proba'):
                y_prob = model_instance.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                metrics['roc_auc'] = round(roc_auc, 4)

                fig, ax = plt.subplots(figsize=(8, 6))
                RocCurveDisplay.from_predictions(y_test, y_prob, name=f"{selected_model} (AUC = {roc_auc:.2f})", ax=ax)
                ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.50)')
                ax.set_title('ROC Curve')
                plots['roc_curve_image'] = generate_plot_image(fig)
        
        elif selected_model == 'random_forest_regressor':
            model_instance = RandomForestRegressor(random_state=random_state)
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)
            metrics['model_type'] = 'Regression'
            metrics['mse'] = round(mean_squared_error(y_test, y_pred), 4)
            metrics['r2_score'] = round(r2_score(y_test, y_pred), 4)
            metrics['message'] = f"Random Forest Regressor trained successfully for '{target_variable}'."

            # Residual Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=y_pred, y=(y_test - y_pred), ax=ax)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('Residual Plot')
            plots['residual_plot_image'] = generate_plot_image(fig)

        elif selected_model == 'kmeans':
            cluster_features = X.columns.tolist() # Use all current features in X for clustering
            X_cluster = X.copy() # Use the pre-split X (features only) for clustering
            
            if X_cluster.empty:
                 return jsonify({"status": "error", "message": "No complete numerical features remaining for KMeans clustering after dropping missing values."}), 400
            
            # For simplicity, hardcode n_clusters or provide a default if not explicitly sent from frontend.
            # In a real app, elbow method or user input would determine n_clusters.
            n_clusters = min(3, X_cluster.shape[0]) # Ensure n_clusters is not greater than number of samples
            if n_clusters < 2:
                return jsonify({"status": "error", "message": f"KMeans requires at least 2 clusters and samples. Only {X_cluster.shape[0]} samples are available."}), 400


            model_instance = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10) # n_init for modern sklearn
            model_instance.fit(X_cluster)
            
            # Assign cluster labels back to the original dataframe for potential future use or visualization
            # This is complex when X is a subset. Let's return labels, and frontend can manage if needed.
            metrics['cluster_labels_preview'] = model_instance.labels_[:5].tolist() # Send first 5 labels as preview

            metrics['model_type'] = 'Clustering'
            metrics['n_clusters'] = n_clusters
            try:
                # Silhouette score requires at least 2 clusters and >= 2 samples per cluster
                if len(np.unique(model_instance.labels_)) > 1 and X_cluster.shape[0] >= 2 and len(np.unique(model_instance.labels_)) < X_cluster.shape[0]:
                    metrics['silhouette_score'] = round(silhouette_score(X_cluster, model_instance.labels_), 4)
                else:
                    metrics['silhouette_score'] = "N/A (insufficient clusters or samples to calculate silhouette score)"
            except Exception as e:
                 metrics['silhouette_score'] = f"Error calculating: {str(e)}"
            
            metrics['message'] = f"K-Means clustering performed successfully with {n_clusters} clusters."

        else:
            return jsonify({"status": "error", "message": f"Invalid model selected: {selected_model}."}), 400

        # Include plots in the response
        metrics['plots'] = plots
        return jsonify({"status": "success", "message": metrics.pop('message'), "model_metrics": metrics}), 200

    except Exception as e:
        print(f"Error training model: {e}")
        return jsonify({"status": "error", "message": f"Error training model: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
