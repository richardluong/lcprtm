from gensim import corpora, models, similarities
import logging, gensim, bz2
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

lda = models.LdaModel.load('tmp/model_2.lda')
dictionary = corpora.Dictionary.load('tmp/dictionary.dict')
# lda.print_topics(100)
# print lda.print_topic(0)

for i in lda.show_topics(num_words=20):
    print i.encode('utf-8')