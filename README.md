# Twitter Sentiment Analysis Project
=====================================

This project uses machine learning to analyze the sentiment of tweets, classifying them as either positive or negative. It utilizes Python with libraries such as `sklearn`, `pandas`, and `matplotlib` for data processing and visualization.

## Project Description
--------------------

The goal of this project is to demonstrate a basic sentiment analysis tool using a Linear Support Vector Machine (SVM) classifier. The model is trained on a dataset of labeled tweets and then used to predict the sentiment of new, unseen tweets.

### Key Features:
- **Data Preprocessing**: Removes mentions, URLs, special characters, and numbers from tweets.
- **Model Training**: Trains an SVM classifier on TF-IDF features.
- **Sentiment Prediction**: Analyzes new tweets and predicts their sentiment.
- **Visualization**: Displays a pie chart showing the distribution of positive and negative sentiments.

## Requirements
------------

To run this project, you need:
- **Python 3.x**
- **Pandas** (`pip install pandas`)
- **Matplotlib** (`pip install matplotlib`)
- **Scikit-learn** (`pip install scikit-learn`)
- **NLTK** (`pip install nltk`)
- **Re** (built-in Python library)

## Instructions
-------------

1. **Clone the Repository**:
git clone https://github.com/rohitsunkari08/Twitter-Sentiment-Analysis

2. **Install Dependencies**:
pip install pandas matplotlib scikit-learn nltk

3. **Run the Project**:
python main.py

4. **Interpret Results**:
- The project will print the sentiment analysis results for the example tweets.
- A pie chart will display the distribution of positive and negative sentiments.

## Contributing
------------
Contributions are welcome! Feel free to enhance the model, add more features, or improve the documentation.

## Example Use Case
-----------------
To analyze new tweets, simply add them to the `tweets` list in the `main.py` file and run the script again.

tweets = [
"I had a great time at the concert",
"The movie was boring",
"Love the new restaurant downtown",
"Hate the traffic in this city",
"The hotel room was very clean",
"The food was terrible at the restaurant",
"I'm so excited for the weekend",
"I'm feeling really stressed today",
"The new iPhone is amazing",
"The new game is really boring"
]
This project serves as a starting point for more advanced sentiment analysis tasks, such as using pre-trained models or exploring other machine learning algorithms.

## Acknowledgments
----------------
- Rohit Sunkari (https://github.com/rohitsunkari08)

## Future Work
-------------
- Explore using more advanced machine learning models.
- Integrate with real-time tweet data.
- Enhance data preprocessing techniques.
