# Juno Model

## Description
Project for training a cryptocurrency price prediction model using historical data and LSTM networks. It leverages libraries like TensorFlow, Pandas, and Scikit-learn for data manipulation, normalization, and model training.

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

## Additional Information
Models have been trained for the following cryptocurrencies:
- BTC
- ETH
- BNB
- XRP
- DOGE
- LINK
- SOL

More models will be trained in the near future.

## Notes
- An internet connection is recommended to download historical cryptocurrency data.
- The model and scaler are saved in the `models/saved_models` folder.

## License
This project is available under the MIT License.