#!/usr/bin/python
# -*- coding: utf-8 -*-

n_best_list_file_name = "test.output.2.nbest2"
input_file = "test.input.tok.1"

class MyNBestList(object):
    def __iter__(self):
        for line in open(n_best_list_file_name):
            yield line.strip().lower()

class MyInput(object):
    def __iter__(self):
        for line in open(input_file):
            yield line.strip().lower()

n_best_list = MyNBestList()
input_list = MyInput()
iter_nbest = iter(n_best_list)

dictonary = {}

target_sentence = iter_nbest.next()

for n, input_sentence in enumerate(input_list):
	input_words = input_sentence.strip().split(" ")

	while (n == int(target_sentence.split("|||")[0].strip())):
		print "Getting phrases from {}th source sentence".format(n)

		alignments = target_sentence.split("|||")[1].split("|")[1:-1:2]
		target_phrases = target_sentence.split("|||")[1].split("|")[0:-1:2]

		for target_phrase, alignment in zip(target_phrases, alignments):
			alignment = alignment.split("-")
			source_phrase = " ".join(input_words[int(alignment[0]):int(alignment[1])+1])

			t = (source_phrase, target_phrase)

			if t in dictonary:
				dictonary[t] += 1
			else:
				dictonary[t] = 1
		try:
			s = iter_nbest.next()
		except StopIteration:
			break

f = open("phrase_pairs.txt", "w+")
for key,value in dictonary.iteritems():
	f.write("'{}','{}':{}\n".format(key[0], key[1], value))




