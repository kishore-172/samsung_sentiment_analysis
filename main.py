import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from textblob import TextBlob
import seaborn as sns
import transformers
from transformers import pipeline
from collections import Counter
import plotly.graph_objects as go
# Suppress the pandas configuration warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")




# Load the image from a local file path
# Load the image logo from a local file path
# Load the pre-trained emotion analysis model


# Set the page configuration
st.set_page_config(layout="wide")

# Load the image logo from a local file path
image_path = "Samsung_Logo_1993.png"

# The URL you want to redirect to when the image is clicked
website_url = "https://www.samsung.com/in/smartphones/?product1=sm-f936bzkdinu&product2=sm-f721bzaainu&product3=sm-s908edrginu"  # Replace with your desired URL

# Display the image logo in the sidebar
st.sidebar.image(image_path, width=150)

# Handle the redirection
with st.sidebar.expander("Visit Website", expanded=False):
    st.markdown(f'<a href="{website_url}" target="_blank">{website_url}</a>', unsafe_allow_html=True)


#title
# # Load the image logo from a local file path
logo_path = "s23_images.jpeg"

# Create a sidebar layout with two columns
logo_col, title_col = st.columns([1, 4])


# Display the image logo in the first column
logo_col.image(logo_path, width=200)

# Display the title in the second column
app_title = "Sentiment Analysis"
title_col.title(app_title)




st.markdown('         This application is all about sentiment analysis of samsung S23 series.')
#sidebar
st.sidebar.title('Sentiment analysis of Samsung Galaxy S23 series')
# sidebar markdown

#loading the data (the csv file is in the same folder)
#if the file is stored the copy the path and paste in read_csv method.
data=pd.read_csv('Cleaned_reviewsData (1).csv')
df=data.copy()

#checkbox to show data
if st.checkbox("Show Data"):
    st.write(data.head(50))


select=st.sidebar.selectbox('Visualisation Of Tweets',['Histogram','Pie Chart'],key=1)
sentiment=data['Rating'].value_counts()
sentiment=pd.DataFrame({'Sentiment':sentiment.index,'reviews':sentiment.values})
st.markdown("###  Sentiment count")
if select == "Histogram":
        fig = px.bar(sentiment, x='Sentiment', y='reviews', color = 'reviews', height= 500)
        st.plotly_chart(fig)
else:
        fig = px.pie(sentiment, values='reviews', names='Sentiment')
        st.plotly_chart(fig)


emotion_analyzer=pipeline("sentiment-analysis")
# Sentiment Analysis using TextBlob
df['Sentiment_Polarity'] = df['User_Review'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['Sentiment'] = df['Sentiment_Polarity'].apply(lambda x: 'Positive' if x > 0.3 else ('Negative' if x < 0 else 'Neutral'))
positive_reviews = ' '.join(df[df['Sentiment'] == 'Positive']['User_Review'])
negative_reviews = ' '.join(df[df['Sentiment'] == 'Negative']['User_Review'])

# df['emotion_lable']=df.User_Review.apply(lambda x: emotion_analyzer(x)[0]['label'])
# df['emotion_score']=df.User_Review.apply(lambda x:emotion_analyzer(x)[0]['score'])
# st.write(df.head())

@st.cache_data()
def clear_cache():
    return None

clear_cache()

# Load the pre-trained emotion analysis model



# @st.cache_data(ttl=60*60)
# def load_emotion_analyzer():
#     model = pipeline("sentiment-analysis")
#     return model




st.subheader('Sentiment Prediction')
user_input = st.text_input("Enter your text:")

if user_input:
    # Perform sentiment analysis using the loaded model
    emotion, score = emotion_analyzer(user_input)[0]['label'], emotion_analyzer(user_input)[0]['score']

    st.write(f"Predicted Emotion: {emotion}")
    st.write(f"Confidence Score: {score:.2f}")
    st.write(f"Polarity score:{TextBlob(user_input):.2f)}")


st.set_option('deprecation.showPyplotGlobalUse', False)

#
# # Review Titles Analysis
# review_title_freq = pd.Series(' '.join(df['Review_Title']).lower().split()).value_counts()
# # Visualize the most common words or phrases in review titles
def plot_sentiment_distribution():
    plt.figure(figsize=(4, 4))
    sns.countplot(df['Sentiment'])
    plt.title('Sentiment distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    return plt.gcf()

def plot_average_word_sentiment():
    plt.figure(figsize=(4, 4.2))
    sns.barplot(x='model_name', y='Sentiment_Polarity', data=df)
    plt.title('Average Word Sentiment by Model Name')
    plt.xlabel('Model Name')
    plt.ylabel('Average Word Sentiment')
    return plt.gcf()

def plot_word_count_distribution():
    df['Word_Count'] = df['User_Review'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(4, 3.5))
    plt.hist(df['Word_Count'],bins=20,density=True)
    plt.title('Distribution of Word Count')
    plt.xlabel('Word Count')
    plt.ylabel('Count')
    return plt.gcf()

# st.markdown("Sentiment and Word Count Analysis")
st.markdown("<h2 style='font-size: 14px; font-weight: bold;;'>Sentiment and Word Count Analysis</h2>", unsafe_allow_html=True)


    # Assuming you have a DataFrame `df` containing relevant columns

    # Create a sidebar layout with three columns
col1, col2, col3 = st.columns(3)

# Visualize Sentiment Distribution
with col1:
    st.pyplot(plot_sentiment_distribution())

# Average Word Sentiment by Model Name
with col2:
    st.pyplot(plot_average_word_sentiment())

# Word Count Analysis
with col3:
    st.plotly_chart(plot_word_count_distribution())



#word clouds


# Display the plots side by side using Streamlit columns
col1, col2 = st.columns(2)
with col1:

    st.markdown("<h2 style='font-size: 14px; font-weight: bold;text-align: center;'>Word frequency of Positive reviews</h2>",
                unsafe_allow_html=True)
    ig1, ax1 = plt.subplots(figsize=(8, 4))
    positive_word_freq = Counter(positive_reviews.lower().split())
    # Visualize the most common words
    plt.figure(figsize=(10, 6))
    wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
        positive_word_freq)
    plt.imshow(wordcloud_positive, interpolation='bilinear')
    # plt.title('Word Cloud for Positive Reviews')
    plt.axis('off')
    plt.show()
    st.pyplot()

with col2:
    st.markdown("<h2 style='font-size: 14px; font-weight: bold;text-align: center;'>Word frequency of Negative reviews</h2>",
                unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    negative_word_freq = Counter(negative_reviews.lower().split())
    # Visualize the most common words
    plt.figure(figsize=(10, 6))
    wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
        negative_word_freq)
    plt.imshow(wordcloud_negative, interpolation='bilinear')
    # plt.title('Word Cloud for Negative Reviews')
    plt.axis('off')
    plt.show()
    st.pyplot()
# Correlation Analysis
# correlation_matrix = df.corr()
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title('Correlation Matrix')
# plt.show()
# st.pyplot()






