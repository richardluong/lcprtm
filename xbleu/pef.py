import numpy as np

n_best_list_file_name = "test.output.2.nbest2"

class MyNBestList(object):
    def __iter__(self):
        for line in open(n_best_list_file_name):
            yield (int(line.split("|||")[0].strip()), float(line.split("|||")[3].strip()))

n_best_list = MyNBestList()
scores = {}

for (i, score) in n_best_list:
	if i in scores:
		scores[i].append(score);
	else:
		s = [score]
		scores[i] = s
	# print "{} {}".format(i, score)

for key, value in scores.iteritems():
	for score in value:
		pef = np.exp(score) / sum(map(lambda x: np.exp(x), value))
		print pef

# print scores