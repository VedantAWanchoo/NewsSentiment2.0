# Merged & Streamlined News Sentiment App

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime, timedelta
from collections import Counter
import nltk
from nltk.data import find
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers.models.bert import BertForSequenceClassification
from transformers import BertForSequenceClassification, BertTokenizer, pipeline
import anthropic
import torch

# Download necessary NLTK data
#nltk.download('punkt')
#nltk.download('stopwords')

try:
    find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')



def load_model():
    """
    Loads the FinBERT model
    """
    device = torch.device("cpu")
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3).to(device)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    classifier = pipeline("text-classification",model=finbert,tokenizer=tokenizer,device=-1)
                          
    return classifier


def predict_sentiment(classifier, text):
    with torch.no_grad():
        outputs = classifier(text, return_all_scores=True)[0]

    # Find the label with highest score manually
    best = max(outputs, key=lambda x: x['score'])
    label = best['label']
    score = best['score']

    if label == 'Positive':
        return 'Positive', score, 'green'
    elif label == 'Negative':
        return 'Negative', score, 'red'
    else:
        return 'Neutral', score, 'gray'


def fetch_news(company, classifier, numberOfArticles=50, selected_sources=None):
    news_data = []
    page = 1

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    while len(news_data) < numberOfArticles:
        for source in (selected_sources or ['']):
            query = f"{company}"
            if source and source != "All":
                query += f" source:{source}"

            url = f"https://www.google.com/search?q={query}&tbm=nws&start={(page-1)*10}"
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')

            articles = soup.find_all('div', class_='SoaBEf')

            for article in articles:
                if len(news_data) >= numberOfArticles:
                    break

                headline = article.find('div', class_='MBeuO').text.strip()
                description = article.find('div', class_='GI74Re').text.strip()
                url = article.find('a', class_='WlydOe')['href']
                source_name = article.find('div', class_='MgUUmf').text.strip()
                timestamp = article.find('div', class_='OSrXXb').text.strip()

                date = convert_to_datetime(timestamp)

                text_to_analyze = f"{headline} {description}"
                predicted_sentiment, score, color = predict_sentiment(
                    classifier, text_to_analyze)

                news_data.append({
                    'headline': headline,
                    'description': description,
                    'url': url,
                    'source_name': source_name,
                    'timestamp': date,
                    'predicted_sentiment': predicted_sentiment,
                    'score': score,
                    'color': color
                })

            if len(news_data) >= numberOfArticles:
                break

            time.sleep(random.uniform(1, 3))  # Random delay between requests

        page += 1

    return news_data[:numberOfArticles]

def convert_to_datetime(relative_time):
    now = datetime.now()
    if 'minute' in relative_time or 'hour' in relative_time:
        return now.date()
    elif 'day' in relative_time:
        days = int(relative_time.split()[0])
        return (now - timedelta(days=days)).date()
    elif 'week' in relative_time:
        weeks = int(relative_time.split()[0])
        return (now - timedelta(weeks=weeks)).date()
    elif 'month' in relative_time:
        months = int(relative_time.split()[0])
        return (now - timedelta(days=months*30)).date()  # Approximation
    elif 'year' in relative_time:
        years = int(relative_time.split()[0])
        return (now - timedelta(days=years*365)).date()  # Approximation
    else:
        return now.date()  # Default to today if parsing fails



def create_word_plot(news_data, company):
    stop_words = set(stopwords.words('english'))
    word_freq = Counter()

    for article in news_data:
        text = f"{article['headline']} {article['description']}"
        #words = word_tokenize(text.lower())

        words = word_tokenize(text.lower(), preserve_line=True)

        words = [word for word in words if word.isalnum() and word not in stop_words and word != company.lower()]
        word_freq.update(words)

    top_words = word_freq.most_common(20)
    df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])

    fig = px.bar(df, x='Word', y='Frequency', title='Most Frequent Words')
    return fig

def create_sentiment_plot(news_data):
    
    df = pd.DataFrame(news_data)
    df['date'] = pd.to_datetime(df['timestamp'])
    df = df.groupby(['date', 'predicted_sentiment']
                    ).size().unstack(fill_value=0).reset_index()

    df_long = df.melt(id_vars='date', var_name='Sentiment', value_name='Count')

    color_mapping = {
        'Positive': 'limegreen',
        'Neutral': 'lightgray',
        'Negative': 'lightcoral'
    }

    fig = px.bar(df_long, x='date', y='Count', color='Sentiment', barmode='stack',
                 color_discrete_map=color_mapping,
                 labels={'date': 'Date', 'Count': 'Number of Articles', 'Sentiment': 'Sentiment'})

    fig.update_layout(
        title='Sentiment over time',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def create_weighted_sentiment_plot(news_data):
    
    df = pd.DataFrame(news_data)
    df['date'] = pd.to_datetime(df['timestamp'])

    # Create a numerical sentiment score
    sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    df['sentiment_score'] = df['predicted_sentiment'].map(sentiment_map)

    # Calculate weighted sentiment and article count for each day
    daily_data = df.groupby('date').agg({
        'sentiment_score': 'mean',
        'predicted_sentiment': 'count'
    }).reset_index()
    daily_data.columns = ['date', 'weighted_sentiment', 'article_count']

    # Create the plot with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add weighted sentiment line
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['weighted_sentiment'],
            mode='lines+markers',
            name='Weighted Sentiment',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ),
        secondary_y=False
    )

    # Add article count line
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['article_count'],
            mode='lines+markers',
            name='Article Count',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ),
        secondary_y=True
    )

    # Update layout
    fig.update_layout(
        title='Weighted Sentiment and Article Count Over Time',
        xaxis_title='Date',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Update y-axes
    fig.update_yaxes(
        title_text="Weighted Sentiment",
        range=[-1, 1],
        zeroline=True,
        zerolinecolor='rgba(0,0,0,0.5)',
        zerolinewidth=1,
        secondary_y=False,
        tickformat='.2f'
    )
    fig.update_yaxes(
        title_text="Article Count",
        secondary_y=True,
        tickformat='d'  # Display as integers
    )

    # Ensure grid lines overlap
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', secondary_y=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', secondary_y=True)

    # Add a horizontal line at y=0 for sentiment
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray", secondary_y=False)

    return fig

ANTH_KEY = st.secrets["ANTHROPIC_API_KEY"]



def get_claude_news_summary(company, news_list):
    client = anthropic.Anthropic(api_key=ANTH_KEY)

    prompt = "Summarise the sentiment of the company "
    prompt += company
    prompt += ", as if summarising for an investor deciding whether to invest in the company or not based on the following news articles: "
    for news_article in news_list:
        prompt += news_article

    prompt += ". Analyse and give the summary in 400 words or less."

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=1000
    )

    summary = response.content[0].text

    return summary

@st.cache_resource
def get_model():
    return load_model()

def display_news_sentiment2():

    st.write("App loaded successfully ✅")

    st.title("Recent News Sentiment Analyzer")

    classifier = get_model()

    sources = [
        'All',
        'Australian Financial Review',
        'Associated Press',
        'ABC News',
        'BBC News',
        'CNN',
        'Financial Post',
        'Fox News',
        'Bloomberg',
        'Business Insider',
        'NBC News',
        'Reuters',
        'Google News',
        'The Wall Street Journal',
        'Newsweek',
        'TechCrunch',
        'TechRadar',
        'The Washington Post'
    ]

    # Input for company name
    company = st.text_input("Enter a company name:", "Microsoft")
    selected_sources = st.multiselect(
        "Select news sources:", sources, default=['All'])

    if st.button("Analyze"):
        with st.spinner("Fetching and analyzing news articles..."):
            news_data = fetch_news(company, classifier, 50, selected_sources)

        if not news_data:
            st.error("No data found or there was an error fetching the news.")
        else:
            news_list = []
            for article in news_data:
                news_list.append(article["headline"])
            claude_summary = get_claude_news_summary(company, news_list)
            st.write(claude_summary)

            # CSS
            st.markdown("""
            <style>
            .card {
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                margin: 10px 0;
                padding: 12px;
                transition: transform 0.2s;
                font-size: 0.9em;
            }
            .card:hover {
                transform: scale(1.01);
            }
            .card-header {
                font-size: 1em;
                margin-bottom: 6px;
                font-weight: bold;
            }
            .card-source-timestamp {
                font-size: 0.8em;
                color: gray;
                margin-bottom: 8px;
            }
            .card-description {
                font-size: 0.9em;
                margin-bottom: 8px;
            }
            .card-sentiment {
                font-size: 0.9em;
                font-weight: bold;
            }
            </style>
            """, unsafe_allow_html=True)

            # Display word plot
            word_plot = create_word_plot(news_data, company)
            st.plotly_chart(word_plot)

            # Display histogram plot
            sentiment_plot = create_sentiment_plot(news_data)
            st.plotly_chart(sentiment_plot)

            # Display sentiment plot
            weighted_sentiment_plot = create_weighted_sentiment_plot(news_data)
            st.plotly_chart(weighted_sentiment_plot, use_container_width=True)

            st.write(
                f"**Showing {len(news_data)} news articles for {company}**")

            # Display news cards
            for article in news_data:
                st.markdown(f"""
                <div class="card" style="border: 1px solid {article['color']};">
                    <div class="card-header" style="color: {article['color']};">
                        <a href="{article['url']}" target="_blank" style="text-decoration: none; color: inherit;">{article['headline']}</a>
                    </div>
                    <div class="card-source-timestamp">
                        {article['source_name']} - {article['timestamp']}
                    </div>
                    <div class="card-description">
                        {article['description']} <a href="{article['url']}" target="_blank">Read more</a>
                    </div>
                    <div class="card-sentiment" style="color: {article['color']};">
                        Predicted Sentiment: {article['predicted_sentiment']}
                    </div>
                </div>
                """, unsafe_allow_html=True)


# & "C:/Users/Vedant Wanchoo/anaconda3/python.exe" -m pip install streamlit pandas plotly requests beautifulsoup4 nltk transformers torch anthropic


# ✅ Run the app
if __name__ == "__main__":
    display_news_sentiment2()
