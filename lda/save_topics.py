from gensim import corpora, models, similarities
import logging, gensim, bz2
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

lda = models.LdaModel.load('data/model.lda')
dictionary = corpora.Dictionary.load('data/dictionary.dict')

f = open("data/weight_initialization.txt", "w+")
i = 0

topics = lda.show_topics(formatted = False, num_topics=lda.num_topics, num_words=len(dictionary))
for topic in topics:
	topic.sort(key=lambda tup: int(tup[1]))
	string = ""
	i += 1
	print "Writing topic #" + str(i)
	for word in topic:
		string += str(word[0]) + " "
	f.write(string + "\n")

