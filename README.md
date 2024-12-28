# AutoML Pipeline Builder

AutoML Pipeline Builder is a Streamlit-based web application that allows users to automate the machine learning pipeline, from dataset upload to model training, evaluation, and export. The tool supports various stages of the machine learning workflow, including exploratory data analysis (EDA), data preprocessing, and model comparison. It caters to both regression and classification tasks, providing an intuitive interface for users to build, evaluate, and export ML models.

---
## Structure
├── app.py

├── modules/

│   ├── dataset.py

│   ├── eda.py

│   ├── preprocessing.py

│   ├── training.py

├── requirements.txt

├── README.md

├── data/

├── automl_env/

├── results/

├── tests/

├── models/

## Features

1. **Dataset Management**
   - Upload datasets in CSV or Excel formats.
   - Preview uploaded data with basic details such as shape, columns, and types (numerical/categorical).

2. **Exploratory Data Analysis (EDA)**
   - View summary statistics for numerical and categorical data.
   - Visualize data distributions, histograms, boxplots, correlation heatmaps, and pairplots.

3. **Data Preprocessing**
   - Encode categorical features using strategies like OneHot, Label, or Drop.
   - Handle missing values with options for mean, median, or most-frequent imputation.
   - Scale numerical features with StandardScaler or MinMaxScaler.
   - Perform train-test splits with customizable test size and random state.

4. **Model Training and Evaluation**
   - Choose between regression and classification tasks.
   - Train various models:
     - Regression: Linear Regression
     - Classification: Logistic Regression, Decision Tree, SVM, Naive Bayes, KNN, Random Forest, Gradient Boosting, AdaBoost
   - Evaluate models with appropriate metrics:
     - Regression: Mean Absolute Error (MAE), Mean Squared Error (MSE), R² score.
     - Classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
   - Visualize results with confusion matrices and ROC curves.

5. **Model Comparison**
   - Compare multiple models on various metrics.
   - Display results in a tabular format for easy decision-making.

6. **Model Export**
   - Save trained models for future use.
   - Export models in a `.joblib` format for integration into production systems.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Streamlit
- Required Python packages (listed in `requirements.txt`)

### Steps to Set Up
1. Clone the Project:
- git clone https://github.com/attaquarks/AutoML.git
- cd AutoML

2. Set Up a Virtual Environment:
- python -m venv automl_env
- source automl_env/bin/activate   # On Linux/Mac
- automl_env\Scripts\activate      # On Windows

3. Install Dependencies:
- pip install -r requirements.txt
- Run the Application: Start the Streamlit application.

4. Run the Application:
- streamlit run app.py
