import sbleu

n_best_list_file_name = "test.output.2.nbest2"
input_file = "test.reference.tok.1"

class MyNbestList(object):
    def __iter__(self):
        for line in open(n_best_list_file_name):
            yield line.strip().lower()

class MyInput(object):
    def __iter__(self):
        for line in open(input_file):
            yield line.strip().lower()

def get_sentence(s):
	return "".join(s.split("|||")[1].split("|")[0:-1:2]).strip().replace("  ", " ").lower()

f = open("sBleu.txt", "w+")
i = 0

n_best_list = MyNbestList()
input_list = MyInput()
iter_n_best = iter(n_best_list)

hypothesis = iter_n_best.next()

for n, reference in enumerate(input_list):

    while (n == int(hypothesis.split("|||")[0].strip())):
        hypothesis = get_sentence(hypothesis)
        sBleu = sbleu.bleu(hypothesis, reference, 3) 
        # print "{}\n{}\n{}\n{}".format(i, hypothesis, reference, sBleu)
        f.write("{}\n".format(sBleu))
        # i += 1
        try:
            hypothesis = iter_n_best.next()
        except StopIteration:
            break

