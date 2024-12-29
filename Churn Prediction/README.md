# Churn Prediction Model

## Project Overview

The **Churn Prediction Model** is designed to predict customer churn using machine learning techniques. It leverages preprocessed data, encoders, and a trained neural network model to classify customers who are likely to leave (churn). The project also supports scalability with an easily deployable architecture.

This project is inspired by the tutorials and guidance provided by **Krish Naik**, a renowned data science YouTuber.

## Technologies Used

- **Python**: Core programming language.
- **TensorFlow/Keras**: For building and running the deep learning model.
- **Pandas**: For data manipulation.
- **Scikit-learn**: For preprocessing and feature encoding.
- **Pickle**: For saving and loading encoders and scalers.
- **Jupyter Notebook**: For experimentation and analysis.

## Project Structure

```
|-- app.py               # Python script for predictions
|-- Churn_Modelling.xlsx # Dataset for churn analysis
|-- experiments/         # Jupyter notebooks for data exploration and experiments
|-- label_encoder_gender.pkl   # Pretrained label encoder for gender
|-- onehot_encoder_geo.pkl     # Pretrained one-hot encoder for geography
|-- model.h5             # Trained neural network model
|-- prediction.ipynb     # Notebook for predictions
|-- requirements.txt     # List of project dependencies
|-- scaler.pkl           # Pretrained scaler for numerical features
```

## Installation

To set up the project, follow these steps:

1. **Clone the repository** (if applicable):

   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. **Install required packages**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify all necessary files are in place**:
   Ensure that the following files exist:
   - `Churn_Modelling.xlsx`
   - `model.h5`
   - `scaler.pkl`
   - `label_encoder_gender.pkl`
   - `onehot_encoder_geo.pkl`

## Usage

1. **Data Preparation**:
   - Place your dataset in the project directory (`Churn_Modelling.xlsx`).

2. **Run Predictions**:
   - Open the `prediction.ipynb` notebook.
   - Execute the cells to preprocess the data, load the model, and make predictions.

   Alternatively, you can use the `app.py` script:

   ```bash
   python app.py
   ```

3. **Interpreting Results**:
   - The model outputs a prediction indicating whether a customer will churn (1) or not (0).

## Functionality

- **Data Preprocessing**:
  - Encodes categorical variables using pre-trained encoders (`label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`).
  - Scales numerical features using `scaler.pkl`.

- **Model**:
  - A neural network model stored in `model.h5`.

- **Prediction**:
  - Predicts churn probabilities for each customer in the dataset.

## Dataset

The dataset used for this project is expected to follow a structure similar to `Churn_Modelling.xlsx`, which includes columns like:
- Customer ID
- Gender
- Geography
- Credit Score
- Age
- Balance
- Tenure
- Exited (Target column indicating churn).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please create an issue or submit a pull request if you have suggestions or improvements.
