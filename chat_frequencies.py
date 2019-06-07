# -*- coding: utf-8 -*-

import pandas as pd 
import re 
import nltk
import string

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')

#prepare what to remove from the text
stopwords = nltk.corpus.stopwords.words('spanish')
#put here words you want to exclude from Stopwords
stopwords_delete = ["no"]
#put here words you want to include in Stopwords
stopwords_add = ['algo', 'alguna', 'alguno', 'algun', 'aquella', 'aquello', 'aquellas', 'aquellos', "llegar", 'ti', "via", 'se', "rt", 'pero', 'porque', 'está', 'es', 'has', 'más', 'mas',
			 'sea', 'este', 'hay', 'sólo', 'mis', 'desde', 'cada', 'ahí', 'da', 
			 'no', 'nos', 'eres', 'unos', 'soy', 'ni', 'después', 'despues', 'sería',
			 'si', 'ellos', 'estas', 'les', 'sido', 'haya', 'allí', 'mía', 
			 'yo', 'aun', 'aún', 'ese', 'tengo', 'tienes', 'vas', 'tal', 'donde', 'hasta', 
			 'tú', 'algun', 'algún', 'voy', 'estás', 'antes', 'están', 'tenía', 'van',
			 'tal', 'entre', 'dónde', 'esas', 'todo', 'toda', 'todas', 'todos', 
			 'q', 'media', 'tanto', 'cerca', 'había', 'casi', 'habrá', 'ella', 'será', 
			 'estos', 'hemos', 'haber', 'uno', 
			 'aquí', 'unas', 'así', 'estar', 
			 'él', 	'asi', 'ido', 	'mí', 'aqui',  'tus', 'estoy', 'fue', 'tan', 
			 'que', 'su', 'sobre', 'la', 'aunque', 
			 'de', 'con', 'o','del', 'medium', 'mi', 'en', 
			 'e', 'y', 'sin', 'tu', 'le', 'va', 
			 'cuando', 'son', 'ma', 'era', 'esto', 'esta', 
			 'han', 'ser', 'ante', 'ti', 'do', 'cómo', 
			 'el', 'a', 'me', 'un', 'lo', 'por', 'para', 'los', 
			 'las', 'te', 'he', 'se', 'una', 'ha', 'qué', 'como', 'eso', 'al', 'ya', 'algo']

new_stopwords = []
for word in stopwords:
	if word not in stopwords_delete:
		new_stopwords.append(word)
stopwords = new_stopwords
if len(stopwords_add) != 0:
	stopwords += stopwords_add
punctuation = list(string.punctuation)
#add punctuation symbols 
punctuation += ['–', '—', '"', "¿", "¡", "``", "''", "...", '_']
#all stopwords and punctuation
stop = stopwords + punctuation
   
#dictionary with replacements
diacritica = {
    "á": "a",
    "ó": "o",
    "í": "i",
    "é": "e",
    "ú": "u",
    "ü": "u"
}
#regular expressions for normaliztion
j = re.compile(r"j{2,}") #detects the character the occurs two and more times
jaja = re.compile(r'(ja){2,}')
jeje = re.compile(r'(je){2,}')
haha = re.compile(r'(ha){2,}')
a = re.compile(r'a{2,}')
e = re.compile(r'e{3,}')
i = re.compile(r'i{2,}')
o = re.compile(r'o{2,}')
u = re.compile(r'u{2,}')
f = re.compile(r'f{2,}')
h = re.compile(r'h{2,}')
m = re.compile(r'm{2,}')
link = re.compile(r'(https?|http)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]')
#function to set all words to lowerase, remove all non-word items, remove digits, tokenize. 
def TextPreprocess(text):
	text_l = text.lower()
	text_w = re.sub(r'[^\w\s]','',text_l)
	text_n = re.sub(r'[0-9]', '', text_w) 
	tokens = nltk.tokenize.word_tokenize(text_n) 
	return tokens
#read the text file with chat
filename = 'chat_main.txt'
text = open(filename, 'rt', encoding="utf8").readlines()
#form a dataframe (table) from the text using \t as delimiter for columns. 
df = pd.DataFrame([x.split('\t') for x in text])
df = df[[0, 1, 2]]
#rename columns
df.columns = ['Timestamp', 'Name', 'Text']
#replace ', '  with '\t' for further preprocessing
df['Timestamp'] = df['Timestamp'].str.replace(', ','\t')

#separate date and time into two columns
df['Date'], df['Time'] = df['Timestamp'].str.split('\t',1).str
#here is the full table with five columns
df = df[['Timestamp', 'Date', 'Time', 'Name', 'Text']]
#if there some missing values, drop them
df = df.dropna(how='any')
#separate the dataframe by author of messages
Name1 = df.loc[df['Name'] == 'Name1']
Name2 = df.loc[df['Name'] == 'Name2']
#making a list of strings where each string is a message
Name1_list = Name1['Text'].tolist()
Name2_list = Name2['Text'].tolist()
#collect all tokens
tokens_Name1 = [] #all tokens
tokens_line_Name1 = [] #lines tokenized
for l in Name1_list: 
	#replace links and media message with a specific token (it shows who sends more fotos and videos, and links)
	l = re.sub(link, 'linksent', l)
	l = re.sub('<Media omitted>', 'fotosent', l)
	#preprocess all tokens
	tokens = TextPreprocess(l)
	token_line = [] #line tokenized
	for t in tokens: 
		#remove all stopwords and punctuation
		if t not in stop: 
			tokens_Name1.append(t)
		token_line.append(t)
	tokens_line_Name1.append(token_line)

tokens_Name2 = []
tokens_line_Name2 = []
for l in Name2_list: 
	l = re.sub(link, 'linksent', l)
	l = re.sub('<Media omitted>', 'fotosent', l)
	tokens = TextPreprocess(l)
	token_line = []
	for t in tokens:
		if t not in stop: 
			tokens_Name2.append(t)
		token_line.append(t)
	tokens_line_Name2.append(token_line)

#remove diacritics to normaize the text and do not count with spelling errors			
tokens_preproc_Name1	= [] #all tokens normalized		
for t in tokens_Name1: 
	#replace the letters with diacritics by the letters without it
	t = t.translate({ord(k): v for k, v in diacritica.items()})
	#replace defined regular expression with single letters
	t = re.sub(j, 'j', t)
	t = re.sub(jaja, 'jaja', t)
	t = re.sub(jeje, 'jaja', t)
	t = re.sub(haha, 'jaja', t)
	t = re.sub(a, 'a', t)
	t = re.sub(e, 'e', t)
	t = re.sub(i, 'i', t)
	t = re.sub(o, 'o', t)
	t = re.sub(u, 'u', t)
	t = re.sub(f, 'f', t)
	t = re.sub(h, 'h', t)
	t = re.sub(m, 'm', t)
	tokens_preproc_Name1.append(t)
	
tokens_line_preproc_Name1 = [] #lines with tokens normalized
for l in tokens_line_Name1: 
	token_line = []
	for t in l: 
		t = t.translate({ord(k): v for k, v in diacritica.items()})
		t = re.sub(j, 'j', t)
		t = re.sub(jaja, 'jaja', t)
		t = re.sub(jeje, 'jaja', t)
		t = re.sub(haha, 'jaja', t)
		t = re.sub(a, 'a', t)
		t = re.sub(e, 'e', t)
		t = re.sub(i, 'i', t)
		t = re.sub(o, 'o', t)
		t = re.sub(u, 'u', t)
		t = re.sub(f, 'f', t)
		t = re.sub(h, 'h', t)
		t = re.sub(m, 'm', t)
		token_line.append(t)
	tokens_line_preproc_Name1.append(token_line)

#add new column to dataset		
Name1['Text_pre'] = tokens_line_preproc_Name1

tokens_preproc_Name2 = []
for t in tokens_Name2: 
	t = t.translate({ord(k): v for k, v in diacritica.items()})
	t = re.sub(j, 'j', t)
	t = re.sub(jaja, 'jaja', t)
	t = re.sub(jeje, 'jaja', t)
	t = re.sub(haha, 'jaja', t)
	t = re.sub(a, 'a', t)
	t = re.sub(e, 'e', t)
	t = re.sub(i, 'i', t)
	t = re.sub(o, 'o', t)
	t = re.sub(u, 'u', t)
	t = re.sub(f, 'f', t)
	t = re.sub(h, 'h', t)
	t = re.sub(m, 'm', t)
	tokens_preproc_Name2.append(t)

tokens_line_preproc_Name2 = []
for l in tokens_line_Name2: 
	token_line = []
	for t in l: 
		t = t.translate({ord(k): v for k, v in diacritica.items()})
		t = re.sub(j, 'j', t)
		t = re.sub(jaja, 'jaja', t)
		t = re.sub(jeje, 'jaja', t)
		t = re.sub(haha, 'jaja', t)
		t = re.sub(a, 'a', t)
		t = re.sub(e, 'e', t)
		t = re.sub(i, 'i', t)
		t = re.sub(o, 'o', t)
		t = re.sub(u, 'u', t)
		t = re.sub(f, 'f', t)
		t = re.sub(h, 'h', t)
		t = re.sub(m, 'm', t)
		token_line.append(t)
	
	tokens_line_preproc_Name2.append(token_line)

Name2['Text_pre'] = tokens_line_preproc_Name2

#stemming
tokens_line_stemmed_Name1 = [] #lines with stemmed words
for l in tokens_line_preproc_Name1:
	stem_line = []
	for t in l:
		t = stemmer.stem(t)
		stem_line.append(t)
	tokens_line_stemmed_Name1.append(stem_line)
	
#add new column with stemmed lines		
Name1['Text_stemmed'] = tokens_line_stemmed_Name1		
		
tokens_line_stemmed_Name2 = []
for l in tokens_line_preproc_Name2:
	stem_line = []
	for t in l:
		t = stemmer.stem(t)
		stem_line.append(t)
	tokens_line_stemmed_Name2.append(stem_line)		

Name2['Text_stemmed'] = tokens_line_stemmed_Name2

print('Calculating frequencies')
#calculate frequent words
freq_N1 = nltk.FreqDist(tokens_preproc_Name1)
freq_N2 = nltk.FreqDist(tokens_preproc_Name2)

#stem all the words to normalize the frequencies
stemmed_N1 = [stemmer.stem(t) for t in tokens_preproc_Name1]
stemmed_N2 = [stemmer.stem(t) for t in tokens_preproc_Name2]
#frequencies of stemmed words
freq_N1_stemmed = nltk.FreqDist(stemmed_N1)
freq_N2_stemmed = nltk.FreqDist(stemmed_N2

most_common_N1 = freq_X.most_common(300)
most_common_stemmed_N1 = freq_N1_stemmed.most_common(300)

most_common_N2 = freq_N2.most_common(300)
most_common_stemmed_N2 = freq_N2_stemmed.most_common(300)	

print('Calculating time and date')
#process the messages in time 
#make aa list of time values
time = list(df.Time.values)
#dictionary to count how many messages are written by hours
time_groups = {}
for i in range(24):
      time_groups[str(i)] = 0  
#counting messages an hour
def add_to_time_groups(current_hour):
      current_hour = str(current_hour)
      time_groups[current_hour] += 1
#take each time value from the list and add it to the dictionary accirdint to the hour
for t in time: 
	#take the first number from the timestamp 
	 current_hour = int(t.split(":")[0])
	 add_to_time_groups(current_hour)
	
#get the list of all dates
month_list = list(df.Date.values)
#changing the date format from 01/01/2018 to 2018/01 (year and month)
month_list_new = []
for m in month_list:
	m = m[6:10]+m[2]+m[3:5]
	month_list_new.append(m)
#dictionary to count messages a month  
month_group = dict([(m, 0) for m in month_list_new])

def add_to_month_group(month):
	month = str(month)
	month_group[month] += 1
	
for m in month_list_new:
	add_to_month_group(m)
#remove if some error appears
month_group.pop("/2011/0")

#plotting the grahics
import matplotlib.pyplot as plt

def plot_graph(time_groups, name):
  plt.bar(range(len(time_groups)), list(time_groups.values()), align='center')
  plt.xticks(range(len(time_groups)), list(time_groups.keys()))
  plt.show()
  
def plot_graph_month(month_group, name):
	plt.bar(range(len(month_group)), list(month_group.values()), align='center') 
	plt.xticks(range(len(month_group)), list(month_group.keys()), rotation=90)
	
name = filename
plot_graph(time_groups, name)	 

plot_graph_month(month_group, name)

#write the tables to .csv files
Name1.to_csv("chat_Name1.csv", encoding='utf-8', index=False)
Name2.to_csv("chat_Name2.csv", encoding='utf-8', index=False)
frames = [Name1, Name2]
df_all = pd.concat(frames)
df_all.to_csv("chat_all.csv", encoding='utf-8', index=False)
  

  
