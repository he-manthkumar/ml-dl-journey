import streamlit as st
import requests
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from urllib.parse import parse_qs, urlparse
from nltk.corpus import stopwords
import nltk
import pandas as pd

st.set_page_config(
    page_title="YouTube Sentiment Lens",
    page_icon="📊",
    layout="wide",
)

@st.cache_resource
def load_stop_words():
    nltk.download('stopwords')
    return set(stopwords.words('english'))


stop_words = load_stop_words()

# Load the mBERT model and tokenizer
@st.cache_resource
def load_model():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

def extract_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.hostname and parsed.hostname.endswith("youtu.be"):
        return parsed.path.lstrip("/") or None
    query = parse_qs(parsed.query)
    return query.get("v", [None])[0]


def fetch_comments(api_key, video_id, max_results=100):
    comments = []
    endpoint = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,
        "textFormat": "plainText",
        "key": api_key,
    }
    next_page_token = None

    while len(comments) < max_results:
        params["maxResults"] = min(100, max_results - len(comments))
        if next_page_token:
            params["pageToken"] = next_page_token
        else:
            params.pop("pageToken", None)
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()

        for item in payload.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"].get("textDisplay", "")
            if comment:
                comments.append(comment)
            if len(comments) >= max_results:
                break

        next_page_token = payload.get("nextPageToken")
        if not next_page_token:
            break

    return comments
def analyze_sentiment(comment):
    inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs.detach().numpy()

st.title("YouTube Comment Sentiment Analysis")

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    api_key = ""

with st.sidebar:
    st.header("Configuration")
    st.caption("Use a valid YouTube Data API v3 key stored in `.streamlit/secrets.toml`.")
    video_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=example")
    max_comments = st.slider("Comments to analyze", min_value=20, max_value=500, step=20, value=100)
    analyze_clicked = st.button("Analyze Comments", type="primary", disabled=not api_key)

if not api_key:
    st.warning("Missing Google API key. Add GOOGLE_API_KEY to .streamlit/secrets.toml and restart the app.")

if analyze_clicked:
    if not api_key:
        st.error("Please provide a valid YouTube Data API key.")
    elif not video_url:
        st.error("Please enter a valid YouTube video URL.")
    else:
        try:
            st.write("Preparing analysis...")
            video_id = extract_video_id(video_url.strip())
            if not video_id:
                st.error("Unable to extract a video ID from the provided URL.")
                st.stop()
            with st.spinner("Fetching comments from YouTube..."):
                comments = fetch_comments(api_key, video_id, max_comments)
            if not comments:
                st.warning("No comments found for this video.")
            else:
                sentiment_scores = [analyze_sentiment(comment) for comment in comments]
                sentiment_counts = np.sum(np.array([score.flatten() for score in sentiment_scores]), axis=0)
                labels = ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars']
                sentiment_df = pd.DataFrame({'Sentiment': labels, 'Count': sentiment_counts})

                total_comments = len(comments)
                total_sentiment = sentiment_counts.sum()
                avg_rating = float(np.dot(sentiment_counts, np.arange(1, 6)) / total_sentiment) if total_sentiment else 0.0
                comment_lengths = [len(comment.split()) for comment in comments]
                avg_length = float(np.mean(comment_lengths)) if comment_lengths else 0.0

                st.success(f"Analyzed {total_comments} comments for video {video_id}.")

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric("Comments analyzed", total_comments)
                metric_col2.metric("Average rating", f"{avg_rating:.2f}")
                metric_col3.metric("Average length", f"{avg_length:.1f} words")

                tab_sentiment, tab_lengths, tab_words = st.tabs([
                    "Sentiment Overview",
                    "Comment Lengths",
                    "Top Words",
                ])

                with tab_sentiment:
                    st.subheader("Sentiment distribution")
                    st.bar_chart(sentiment_df.set_index("Sentiment"), use_container_width=True)
                    fig, ax = plt.subplots()
                    sentiment_percentages = [(count / total_sentiment) * 100 if total_sentiment else 0 for count in sentiment_counts]
                    ax.pie(sentiment_percentages, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
                    ax.set_title('Sentiment proportion')
                    st.pyplot(fig)

                with tab_lengths:
                    st.subheader("Comment length distribution")
                    fig, ax = plt.subplots()
                    ax.hist(comment_lengths, bins=20, color='skyblue', edgecolor='black')
                    ax.set_title('Comment length distribution')
                    ax.set_xlabel('Number of words')
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)

                with tab_words:
                    st.subheader("Most frequent meaningful words")
                    words = [word.lower() for comment in comments for word in comment.split() if word.lower() not in stop_words]
                    word_counts = Counter(words)
                    most_common = word_counts.most_common(10)
                    if most_common:
                        words_labels, counts = zip(*most_common)
                        fig, ax = plt.subplots()
                        ax.barh(words_labels, counts, color='teal')
                        ax.set_xlabel('Frequency')
                        ax.set_ylabel('Words')
                        ax.invert_yaxis()
                        st.pyplot(fig)
                    else:
                        st.info("Not enough non-stop words to display a chart.")

                with st.expander("Preview sample comments"):
                    sample_count = min(10, total_comments)
                    for comment in comments[:sample_count]:
                        st.markdown(f"- {comment}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
