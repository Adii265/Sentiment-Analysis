import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from textblob import Word
import re
from sklearn import preprocessing

class cal():
	a = "aabc"
	b = "aaaaa"
	




	def c(a,b):
		data = pd.read_csv('Historie/text_1.csv')
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



		#Removing Punctuations, Symbols
		data['Content'] = data['Content'].str.replace('[^\w\s]',' ')

		#Removing Stop words using nltk
		stop = stopwords.words('english')
		data['Content'] = data['Content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

		#Lemmatisation
		data['Content'] = data['Content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

		#Correcting letter repitions
		def de_repeat(text):
			pattern = re.compile(r"(.)\1{2,}")
			return pattern.sub(r"\1\1", text)
		data['Content'] = data['Content'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))
		lbl_enc = preprocessing.LabelEncoder()

		# Code to find the top 10,000 rarest words appearing in the data
		freq = pd.Series(' '.join(data['Content']).split()).value_counts()[-10000:]

		# Removing all those rarely appearing words from the data
		freq = list(freq.index)
		data['Content'] = data['Content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

		lbl_enc = preprocessing.LabelEncoder()
		y = lbl_enc.fit_transform(data.Emotion.values)
		print(lbl_enc.classes_)
		


		# Splitting into training and testing data in 90:10 ratio
		from sklearn.model_selection import train_test_split

		X_train, X_val, y_train, y_val = train_test_split(data.Content.values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)

 

		# Extracting Count Vectors Parameters
		from sklearn.feature_extraction.text import CountVectorizer
		count_vect = CountVectorizer(analyzer='word')
		count_vect.fit(data['Content'])
		X_train_count =  count_vect.transform(X_train)
		X_val_count =  count_vect.transform(X_val)

		
		from sklearn.metrics import accuracy_score
		# Model 1: Multinomial Naive Bayes Classifier
		from sklearn.naive_bayes import MultinomialNB
		nb2 = MultinomialNB()
		nb2.fit(X_train_count, y_train)
		y_pred = nb2.predict(X_val_count)
		print('naive bayes count vectors accuracy %s' % accuracy_score(y_pred, y_val))

		
		

		data = pd.DataFrame([b])
		data[0] = data[0].str.replace('[^\w\s]',' ')
		stop = stopwords.words('english')
		data[0] = data[0].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
		data[0] = data[0].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
	
		# Extracting Count Vectors feature from our data
	
		tweet_count = count_vect.transform(data[0])
		q=nb2.predict(tweet_count)
		#print(q)
		'''for i in range(len(data)):
			print("Statement ",data[0][i])
			print("Emotion ",lbl_enc.classes_[q[i]]		

		'''

		return lbl_enc.classes_[q]

'''
tweets = pd.DataFrame([b])
		#print(tweets)
		# Doing some preprocessing on these tweets as done before
		tweets[0] = tweets[0].str.replace('[^\w\s]',' ')
		from nltk.corpus import stopwords
		stop = stopwords.words('english')
		tweets[0] = tweets[0].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
		from textblob import Word
		tweets[0] = tweets[0].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
		# Extracting Count Vectors feature from our tweets
		tweet_count = count_vect.transform(tweets[0])
		#Predicting the emotion of the tweet using our already trained linear SVM
		tweet_pred = nb.predict(tweet_count)
		#print(tweet_pred)
'''