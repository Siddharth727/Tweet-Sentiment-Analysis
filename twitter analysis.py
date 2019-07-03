import csv
from textblob import TextBlob
import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.corpus import stopwords
import string
import nltk 
import textmining
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import numpy as np

os.chdir('S:/Data science/Advance Concepts/Text Mining')

twt = pd.read_csv('twitter training data.csv', encoding = 'latin-1')

twt.head()

twt = twt.iloc[:1000]

#nltk.download()
# Sentiment analysis using Text Blob
# Creating empty dataframe to store results
FinalResults = pd.DataFrame()

# Run Engine
for i in range(0, twt.shape[0]):
    
    blob = TextBlob(twt.iloc[i,5])
    
    temp = pd.DataFrame({'Tweets': twt.iloc[i,5], 'Polarity': blob.sentiment.polarity}, index = [0])
    
    FinalResults = FinalResults.append(temp)  


FinalResults['Sentiment'] = FinalResults['Polarity'].apply(lambda x: 'Positive' if x>0 else 'Negative' if x<0 else 'Neutral')

FinalResults['Sentiment'].describe()

#Results: Most of the tweets are Neutral

# Sentiment Analysis using Vader
FinalResults_Vader = pd.DataFrame()

# Creating engine
analyzer = SentimentIntensityAnalyzer()

# Run Engine
for i in range(0, twt.shape[0]):
    
    snt = analyzer.polarity_scores(twt.iloc[i,5])
    
    temp = pd.DataFrame({'Tweets': twt.iloc[i,5], 'Polarity': list(snt.items())[3][1]}, index = [0])

    FinalResults_Vader = FinalResults_Vader.append(temp)

FinalResults_Vader['Sentiment'] = FinalResults_Vader['Polarity'].apply(lambda x: 'Positive' if x>0 else 'Negative' if x<0 else 'Neutral')

FinalResults_Vader['Sentiment'].describe()

#Results: Most of the tweets are Negative

wordcloud = WordCloud(width = 1000, height = 500, stopwords = STOPWORDS, background_color = 'white').generate(
                        ''.join(twt[:6]))

plt.figure(figsize = (15,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

wordcloud.to_file("tweets.png")