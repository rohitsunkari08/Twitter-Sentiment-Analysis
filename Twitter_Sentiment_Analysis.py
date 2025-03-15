import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Preprocess the text data
def preprocess_text(text):
    # Remove mentions
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Build the sentiment analysis model
def build_model(X_train, y_train):
    # Convert text data to TF-IDF features
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    
    # Initialize Linear SVM classifier
    svm_classifier = LinearSVC()
    
    # Train the model
    svm_classifier.fit(X_train_tfidf, y_train)
    return tfidf_vectorizer, svm_classifier

# Analyze tweet sentiment
def analyze_tweet(tweet, model):
    # Preprocess tweet text
    preprocessed_tweet = preprocess_text(tweet)
    
    # Convert tweet to TF-IDF features using the same vectorizer
    tweet_tfidf = model[0].transform([preprocessed_tweet])
    
    # Predict sentiment
    sentiment = model[1].predict(tweet_tfidf)[0]
    
    return sentiment

# Main function
def main():
    # Training data
    data = {
        "text": [
            "I love this product",
            "This product is terrible",
            "I'm so happy with it",
            "It's really bad",
            "The service was amazing",
            "The food was disgusting",
            "Great experience overall",
            "Worst hotel ever",
            "I'm so excited for the concert",
            "The movie was boring",
            "Love the new restaurant downtown",
            "Hate the traffic in this city",
            "Fantastic customer support",
            "Terrible customer service",
            "I'm really enjoying this book",
            "This book is so boring",
            "The hotel room was clean and cozy",
            "The room was dirty and uncomfortable",
            "I love the new iPhone",
            "The new iPhone is overpriced",
            "Great food at the party",
            "The party was a disaster",
            "I'm so grateful for my friends",
            "I'm feeling really lonely",
            "The weather is perfect today",
            "The weather is awful today",
            "I'm excited for the weekend",
            "I'm dreading the weekend",
            "Love the new TV show",
            "Hate the new TV show",
            "The park is beautiful",
            "The park is dirty",
            "I'm so happy to be home",
            "I'm feeling really homesick",
            "The food at the restaurant was amazing",
            "The food at the restaurant was terrible",
            "I love playing video games",
            "I hate playing video games",
            "The new game is so much fun",
            "The new game is really boring",
            "I'm so excited for the holiday",
            "I'm dreading the holiday",
            "The hotel staff was friendly",
            "The hotel staff was rude",
            "I love the new music album",
            "I hate the new music album",
            "The concert was amazing",
            "The concert was disappointing",
            "I'm so grateful for my family",
            "I'm feeling really stressed",
            "The weather forecast is accurate",
            "The weather forecast is wrong",
            "I love the new fashion trend",
            "I hate the new fashion trend",
            "The city is so clean",
            "The city is really dirty",
            "I'm so happy to be alive",
            "I'm feeling really sad",
            "The new movie is great",
            "The new movie is terrible"
        ],
        "sentiment": [
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
        ]
    }
    
    # Ensure both lists have the same length
    X_train = data["text"][:50]  # Limit to 50 elements
    y_train = data["sentiment"][:50]  # Limit to 50 elements
    
    # Split data into training and test sets
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Build model
    model = build_model(X_train_split, y_train_split)
    
    # Evaluate model on test set
    X_test_tfidf = model[0].transform(X_test_split)
    y_pred = model[1].predict(X_test_tfidf)
    print("Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test_split, y_pred))
    print("Classification Report:\n", classification_report(y_test_split, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test_split, y_pred))
    
    # Example tweets for analysis
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
    
    # Analyze each tweet
    sentiment_results = []
    positive_tweets = []
    negative_tweets = []
    for tweet in tweets:
        sentiment = analyze_tweet(tweet, model)
        sentiment_results.append(sentiment)
        if sentiment == 1:
            positive_tweets.append(tweet)
        else:
            negative_tweets.append(tweet)
    
    # Count positive and negative sentiments
    positive_count = sum(1 for sentiment in sentiment_results if sentiment == 1)
    negative_count = sum(1 for sentiment in sentiment_results if sentiment == 0)
    
    # Print positive and negative tweets
    print("\nPositive Tweets:")
    for tweet in positive_tweets:
        print(tweet)
    
    print("\nNegative Tweets:")
    for tweet in negative_tweets:
        print(tweet)
    
    # Plot sentiment distribution as a pie chart
    labels = ['Negative', 'Positive']
    sizes = [negative_count, positive_count]
    colors = ['blue', 'yellow']  # Changing colors to blue and yellow
    
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title("Sentiment Analysis Results")
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()
