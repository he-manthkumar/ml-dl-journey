# YouTube Comment Sentiment Analysis

## Project Overview

The **YouTube Comment Sentiment Analysis** project analyzes comments from a specified YouTube video URL, utilizing the mBERT (Multilingual BERT) model for sentiment analysis. The collected comments are visualized in various formats to provide insights into user sentiments.

## Technologies Used

- **Python**: Programming language for development.
- **Google API Client**: For accessing the YouTube Data API.
- **Transformers**: Library for utilizing the mBERT model for sentiment analysis.
- **Matplotlib**: For data visualization.
- **WordCloud**: To generate word clouds from comments.

## Installation

To set up the project, follow these steps:

1. **Clone the repository** (if applicable):

   ```bash
   git clone https://github.com/he-manthkumar/myprojects.git
   cd yourrepository
   ```

2. **Install required packages**:

   ```bash
   pip install google-api-python-client transformers torch matplotlib wordcloud
   ```

3. **Obtain YouTube Data API Key**:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/).
   - Enable the YouTube Data API v3 and create API credentials.

## Running the Jupyter Notebook

1. Ensure you have Jupyter Notebook installed. If you don't, you can install it using:

   ```bash
   pip install notebook
   ```

2. Navigate to the directory where your Jupyter Notebook (`your_notebook.ipynb`) is located.

3. Launch the Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open the notebook from the Jupyter interface and run the cells to perform the sentiment analysis.

## Usage

1. **Input the YouTube video URL** when prompted in the notebook:

   ```
   Enter the YouTube video URL: https://www.youtube.com/watch?v=your_video_id
   ```

2. **Enter your API key** in the script:

   ```python
   api_key = "YOUR_API_KEY_HERE"  # Replace with your actual API key
   ```

3. **View the results**: The program will fetch comments, perform sentiment analysis, and display visualizations.

## Functionality

- **Get Comments**: Fetches comments from the provided YouTube video URL using the YouTube Data API.
- **Sentiment Analysis**: Uses mBERT to analyze the sentiment of each comment, scoring from 0 (Very Negative) to 4 (Very Positive).
- **Visualization**: Generates various plots to represent the sentiment analysis results, including:
  - Histogram of sentiment scores
  - Pie chart of sentiment distribution
  - Bar plot of sentiment categories
  - Box plot of sentiment scores
  - Word cloud of comments

## Visualization

The visualizations include:

- **Histogram**: Displays the distribution of sentiment scores.
- **Pie Chart**: Shows the proportion of each sentiment category.
- **Bar Plot**: Illustrates the count of comments in each sentiment category.
- **Box Plot**: Visualizes the spread of sentiment scores.
- **Word Cloud**: Represents frequently occurring words in the comments.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.
