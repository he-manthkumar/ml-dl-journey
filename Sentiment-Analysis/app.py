import streamlit as st
from googleapiclient.discovery import build
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
from collections import Counter
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the mBERT model and tokenizer
@st.cache_resource
def load_model():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# YouTube API function to fetch comments
def fetch_comments(api_key, video_id, max_results=100):
    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )
    response = request.execute()
    comments = []
    for item in response.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)
    return comments

# Function to analyze sentiment
def analyze_sentiment(comment):
    inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs.detach().numpy()

# Create a Word Cloud
def create_wordcloud(text):
    wordcloud = WordCloud(background_color=None, mode="RGBA").generate(text)
    buffer = BytesIO()
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(buffer, format="png", transparent=True)
    buffer.seek(0)
    return buffer

# Streamlit App Interface
st.title("YouTube Comment Sentiment Analysis")

# Predefined API Key
api_key = "AIzaSyACyV2Hn-oV0cpvGGvSyD6p-gZm_kwB4To"

# Input YouTube Video URL
video_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=example")

if st.button("Analyze"):
    if not video_url:
        st.error("Please enter a valid YouTube video URL.")
    else:
        try:
            st.write("Fetching comments...")
            # Extract video ID from the URL
            video_id = video_url.split("v=")[-1].split("&")[0]
            
            # Fetch comments using YouTube API
            comments = fetch_comments(api_key, video_id)
            
            if not comments:
                st.warning("No comments found for this video.")
            else:
                # Combine all comments into a single string
                all_text = " ".join(comments)
                
                # Analyze sentiments
                sentiment_scores = [analyze_sentiment(comment) for comment in comments]
                sentiment_counts = np.sum(np.array([score.flatten() for score in sentiment_scores]), axis=0)
                labels = ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars']
                
                # Word Cloud
                st.write("**Word Cloud:**")
                buffer = create_wordcloud(all_text)
                st.image(buffer, use_column_width=True)
                
                # Sentiment Distribution (Bar Chart)
                st.write("**Sentiment Distribution:**")
                st.bar_chart(sentiment_counts, width=0.8)
                
                # Comment Length Distribution (Histogram)
                comment_lengths = [len(comment.split()) for comment in comments]
                st.write("**Comment Length Distribution:**")
                fig, ax = plt.subplots()
                ax.hist(comment_lengths, bins=20, color='skyblue', edgecolor='black')
                ax.set_title('Comment Length Distribution')
                ax.set_xlabel('Number of Words')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
                
                # Pie Chart of Sentiment Scores
                st.write("**Sentiment Proportion (Pie Chart):**")
                fig, ax = plt.subplots()
                total = sum(sentiment_counts)
                sentiment_percentages = [(count / total) * 100 for count in sentiment_counts]
                ax.pie(sentiment_percentages, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
                ax.set_title('Sentiment Proportion')
                st.pyplot(fig)
                
                # Most Frequent Words (Bar Chart)
                words = [word.lower() for comment in comments for word in comment.split() if word.lower() not in stop_words]
                word_counts = Counter(words)
                most_common = word_counts.most_common(10)
                words, counts = zip(*most_common)
                
                st.write("**Most Frequent Words:**")
                fig, ax = plt.subplots()
                ax.barh(words, counts, color='teal')
                ax.set_xlabel('Frequency')
                ax.set_ylabel('Words')
                ax.invert_yaxis()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred: {e}")
