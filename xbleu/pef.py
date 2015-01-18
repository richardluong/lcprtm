import numpy as np

n_best_list_file_name = "test.output.2.nbest2"

def get_sentence(s):
	return "".join(s.split("|||")[1].split("|")[0:-1:2]).strip().replace("  ", " ").lower()

class MyNBestList(object):
    def __iter__(self):
        for line in open(n_best_list_file_name):
            yield (get_sentence(line), float(line.split("|||")[3].strip()))

def get_pef_dictionary:
	n_best_list = MyNBestList()
	scores = {}
	pef = {}

	for (sentence, score) in n_best_list:
		if sentence in scores:
			scores[sentence].append(score);
		else:
			s = [score]
			scores[i] = s
		# print "{} {}".format(i, score)

	for key, value in scores.iteritems():
		for score in value:
			pef = np.exp(score) / sum(map(lambda x: np.exp(x), value))
			print pef

# print scores