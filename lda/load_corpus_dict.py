from gensim import corpora, models, similarities
import logging, gensim, bz2
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MyCorpus(object):
    def __iter__(self):
        for line in open('corpus.enzh'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())

# # remove common words

# stoplist = set("a about above after again against all am an and any are aren&apos;t as at be because been before being below between both but by can&apos;t cannot could couldn&apos;t did didn&apos;t do does doesn&apos;t doing don&apos;t down during each few for from further had hadn&apos;t has hasn&apos;t have haven&apos;t having he he&apos;d he&apos;ll he&apos;s her here here&apos;s hers herself him himself his how how&apos;s i i&apos;d i&apos;ll i&apos;m i&apos;ve if in into is isn&apos;t it it&apos;s its itself let&apos;s me more most mustn&apos;t my myself no nor not of off on once only or other ought our ours	ourselves out over own same shan&apos;t she she&apos;d she&apos;ll she&apos;s should shouldn&apos;t so some such than that that&apos;s the their theirs them themselves then there there&apos;s these they they&apos;d they&apos;ll they&apos;re they&apos;ve this those through to too under until up very was wasn&apos;t we we&apos;d we&apos;ll we&apos;re we&apos;ve were weren&apos;t what what&apos;s when when&apos;s where where&apos;s which while who who&apos;s whom why why&apos;s with won&apos;t would wouldn&apos;t you you&apos;d you&apos;ll you&apos;re you&apos;ve your yours yourself yourselves ? / ! , . is".split())

# collect statistics about all tokens
dictionary = corpora.Dictionary(line.lower().split() for line in open('corpus.enzh'))
# remove stop words and words that appear only once
# stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
# once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
# dictionary.filter_tokens(once_ids) # remove stop words and words that appear only once
dictionary.compactify()
dictionary.save_as_text('data/dictionary.dict') # store the dictionary, for future reference

corpus = MyCorpus()
corpora.MmCorpus.serialize('data/corpus.mm', corpus)