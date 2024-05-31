# Twitter Entity Sentiment Analysis

This repository contains a project for sentiment analysis of tweets, focusing on entities mentioned in the tweets. The project uses machine learning and deep learning techniques to classify the sentiment of tweets into four categories: Irrelevant, Negative, Neutral, and Positive.

## Dataset

The dataset used in this project is from Kaggle and contains tweets with associated sentiment labels. The dataset is divided into training and validation sets.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/DevSingh28/sentiment_analyser.git
    cd twitter-entity-sentiment-analysis
    ```

2. Download the dataset from Kaggle:
    ```bash
    kaggle datasets download jp797498e/twitter-entity-sentiment-analysis
    unzip twitter-entity-sentiment-analysis.zip -d data/
    ```

## Usage

1. Open the Jupyter notebook:
    ```bash
    jupyter notebook sentiment_analyzer.ipynb
    ```

2. Follow the steps in the notebook to preprocess the data, train the model, evaluate its performance, and make predictions.

3. To predict sentiment for a custom input, use the `predict_sentiment` function defined in the notebook:
    ```python
    user_input = "Have a good day sir"
    predicted_sentiment = predict_sentiment(user_input)
    print(f"The predicted sentiment is: {predicted_sentiment}")
    ```

## Model

The project uses two models for sentiment analysis:
1. **Naive Bayes Model**: A simple yet effective model using TF-IDF vectorization and Multinomial Naive Bayes classifier.
2. **LSTM Model**: A deep learning model using Long Short-Term Memory (LSTM) layers for better performance on sequential data.

### Training the LSTM Model

The LSTM model is trained on the combined entity and tweet content using TensorFlow and Keras. The model's architecture includes:
- Text vectorization using TensorFlow's `TextVectorization` layer.
- Embedding layer to represent words in a dense vector space.
- LSTM layers to capture sequential dependencies.
- Dense layer with softmax activation for multi-class classification.

### Evaluation

The performance of the models is evaluated using:
- Classification report (precision, recall, F1-score).
- Confusion matrix visualization.

## Results

The Naive Bayes model achieved an accuracy of 72%. The LSTM model's performance is further evaluated and improved through training.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any features, enhancements, or bug fixes.

## Acknowledgements

- [Kaggle](https://www.kaggle.com/) for providing the dataset.
- TensorFlow and Keras teams for developing excellent deep learning libraries.


