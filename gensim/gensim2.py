from gensim import corpora, models, similarities
import logging, gensim, bz2
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# dictionary = corpora.Dictionary.load('tmp/dictionary.dict')
corpus = corpora.MmCorpus('tmp/corpus.mm')

lda = models.LdaMulticore(corpus=corpus, num_topics=100)
lda.save('tmp/model_3.lda')