# Juno Model

## Description
Project for training a cryptocurrency and stocks price prediction model using historical data and LSTM networks. It leverages libraries like TensorFlow, Pandas, and Scikit-learn for data manipulation, normalization, and model training.

## Installation
1. Clone the repository.
2. Navigate to the project directory:
    ```
    cd juno-model
    ```
3. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage
- To train the model, run:
  ```
  python main.py
  ```

- You will need an TwelveData API Key.
  ```
  TWELVEDATA_API_KEY=YOUR_TWELVEDATA_API_KEY
  ```


## Notes
- An internet connection is recommended to download historical cryptocurrency data.
- The model and scaler are saved in the `models` folder.

## License
This project is available under the MIT License.