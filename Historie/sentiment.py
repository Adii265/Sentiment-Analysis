import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from textblob import Word
import re
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import pickle



class sentiment_analysis():
	a = "aabc"
	b = "aaaaa"
	
	def preprocessing_algo(a,b):
		#data = pd.read_csv('Historie/text_1.csv')
		data = pd.read_csv('Historie/Book1.csv',low_memory=False)
		data = data.drop(data[data.Emotion == 'boredom'].index)
		data = data.drop(data[data.Emotion == 'enthusiasm'].index)
		data = data.drop(data[data.Emotion == 'empty'].index)
		data = data.drop(data[data.Emotion == 'fun'].index)
		data = data.drop(data[data.Emotion == 'relief'].index)
		data = data.drop(data[data.Emotion == 'surprise'].index)
		data = data.drop(data[data.Emotion == 'hate'].index)
		data = data.drop(data[data.Emotion == 'anger'].index)
		data = data.drop(data[data.Emotion == 'fear'].index) 

		lbl_enc = preprocessing.LabelEncoder()
		y = lbl_enc.fit_transform(data.Emotion.values)
		print(lbl_enc.classes_)
		with open('Historie/sav.pkl', 'rb') as file:
			nb2 = pickle.load(file)
		with open('Historie/sav2.pkl', 'rb') as file:
			count_vect = pickle.load(file)

		data = pd.DataFrame([b])
		data[0] = data[0].str.replace('[^\w\s]',' ')
		stop = stopwords.words('english')
		data[0] = data[0].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
		data[0] = data[0].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
	
		# Extracting Count Vectors feature from our data
	
		tweet_count = count_vect.transform(data[0])
		q=nb2.predict(tweet_count)
		return lbl_enc.classes_[q]
