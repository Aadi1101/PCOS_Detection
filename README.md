# PCOS Detection Using Machine Learning

This project detects Polycystic Ovary Syndrome (PCOS) using machine learning models based on various health metrics. It features a Flask-based web application for easy access and interaction.

## Table of Contents
- [Project Description](#project-description)
- [Prerequisites](#prerequisites)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Input Parameters](#input-parameters)
- [API Endpoints](#api-endpoints)
- [ML Model](#ml-model)
- [Model Tuning and Preprocessing](#model-tuning-and-preprocessing)
- [Feature Selection using Reinforcement Learning](feature-selection-using-reinforcement-learning)
- [Future Enhancements](#future-enhancements)
- [References or Documentation Links](#references-or-documentation-links)
- [Contributing](#contributing)
- [License](#license)

## Project Description
Polycystic Ovary Syndrome (PCOS) is a common hormonal disorder affecting women of reproductive age. This project aims to detect PCOS using machine learning models based on various health and hormonal metrics. The project features a web-based interface powered by Flask.

## Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.7 or higher
- pip (Python package installer)

## Technologies Used
The following technologies are used in this project:

- **Programming Language**: Python 3.7+
- **Framework**: Flask (for building the web application)
- **Machine Learning Libraries**:
  - **scikit-learn**
  - **xgboost**
  - **GridSearchCV** (for hyperparameter tuning)
  - **StandardScaler** (for data normalization)
- **Data Processing Libraries**:
  - **numpy**
  - **pandas**
  - **seaborn**
  - **openpyxl**
- **Serialization**: dill (for serializing and deserializing ML models)
- **Notebook**: Jupyter Notebook (for experimentation)

## Installation

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/pcos-detection.git
   cd pcos-detection
   ```
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use:
   venv\Scripts\activate
   ```
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Flask app:**

   ```bash
   python app.py
   ```
The Flask app will start at http://127.0.0.1:5000/.

## Usage
### Web Application
1. Once the Flask server is running, open your web browser and navigate to http://127.0.0.1:5000/ to access the PCOS detection page.

2. Enter the required health parameters (e.g., Age, Weight, BMI, Blood Group), and submit the form to get a prediction.

### API Usage
Alternatively, you can use the API for predictions by sending a POST request to /predict.

Example:

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"data":"28,65,162,24.8,B+,72,16,12.5,R,28,3,N,0,1.2,2.3,5.5,6.3,0.9,40,30,0.75,2.3,4.1,20.4,4.5,110,N,N,N,N,Y,N,120,80,12,15,18,22,5.5"}'
```
The system will return a prediction on whether PCOS is likely.

## Input Parameters
The system expects the following input parameters (comma-separated or via JSON):

- Age (yrs): Age of the individual in years.
- Weight (Kg): Body weight in kilograms.
- Height (Cm): Height in centimeters.
- BMI: Body Mass Index.
- Blood Group: Blood group type (e.g., A+, B+, O-).
- Pulse rate (bpm): Heart rate in beats per minute.
- RR (breaths/min): Respiratory rate in breaths per minute.
- Hb (g/dl): Hemoglobin level.
- Cycle (R/I): Regular or Irregular menstrual cycle.
- Cycle Length (days): Length of menstrual cycle.
- Marriage Status (yrs): Years since marriage.
- Pregnant (Y/N): Whether the individual is pregnant.
- No. of Abortions: Number of abortions.
- Hormonal Levels: Beta-HCG, FSH, LH, and other hormone metrics.
- Lifestyle Factors: Weight gain, hair growth, skin darkening, pimples, fast food consumption, exercise.
- Blood Pressure: Systolic and Diastolic readings.
- Follicle Count: Left and right ovary follicle count.
- Endometrium Thickness: Measurement of endometrium thickness.

## API Endpoints
1. ```/``` (Home)
- Method: GET
- Description: Displays the homepage where users can input their health data for PCOS detection.
2. ```/predict``` (Prediction)
- Method: POST
- Description: Takes input data via JSON and returns a PCOS detection result.
- Request Payload:
```json
{
  "data": "age,weight,height,bmi,etc."
}
```
- Response: A prediction on whether the individual has PCOS.

## ML Model
- Algorithms:
    - XGBoost: A gradient boosting algorithm for comparison.
    - Model Training: The model was trained using health and hormone metrics relevant to PCOS diagnosis.
    - Model Serialization: The model is saved in a model.pkl file, and dill is used for model serialization.
## Model Tuning and Preprocessing
- Preprocessing:

    - The input data was normalized using StandardScaler to ensure that all features were on the same scale, which is important for algorithms like XGBoost and SVM.

- Hyperparameter Tuning:

    - GridSearchCV was used for hyperparameter optimization to find the best combination of parameters for the machine learning models. The grid search was applied to both SVM and XGBoost models to improve accuracy.
    Example:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [4, 6, 8],
    'n_estimators': [100, 200]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
```
This process was essential in selecting the optimal model parameters and improving prediction performance.

## Feature Selection Using Reinforcement Learning
In addition to traditional machine learning methods, this project employs Reinforcement Learning (RL) for automatic feature selection. By using a Q-learning approach, the model dynamically selects the most relevant features during training. This method balances exploration and exploitation to identify the feature set that maximizes model performance over multiple training episodes.

### Key Components:
- **State Representation**: A binary vector representing selected features.
- **Action Selection**: The RL agent chooses which feature to toggle (select or exclude).
- **Reward**: Generated based on model accuracy.
- **Q-table Update**: The Q-table is updated after each step using the Bellman equation to improve future decisions.
- **Exploration vs Exploitation**: The epsilon-greedy strategy balances between trying new features (exploration) and relying on known effective features (exploitation).

This method helps optimize feature selection, leading to better model performance and potentially reducing overfitting.

## Future Enhancements
- Implement a more detailed user interface for better user experience.
- Integrate additional health metrics for improved prediction accuracy.
- Add user authentication to allow for personalized user experiences.

## References or Documentation Links
- [Flask Documentation](https://flask.palletsprojects.com/en/3.0.x/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Xgboost Documentation](https://xgboost.readthedocs.io/en/stable/python/index.html#)

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Aadi1101/PCOS_Detection/blob/main/LICENSE) file for details.
