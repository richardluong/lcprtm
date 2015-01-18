import sbleu
import linecache

def get_source_sentence_words(source_sentence):
    return source_sentence.strip().split(" ");

def get_phrase_pair_dict_n(source_sentence, target_sentence):

    source_sentence_words = get_source_sentence_words(source_sentence)
    alignments = target_sentence.split("|||")[1].split("|")[1:-1:2]
    target_phrases = target_sentence.split("|||")[1].split("|")[0:-1:2]

    dictonary = {}

    for target_phrase, alignment in zip(target_phrases, alignments):
        # print "Getting alignments and phrase pairs for " + target_sentence 
        alignment = alignment.split("-")
        source_phrase = " ".join(source_sentence_words[int(alignment[0]):int(alignment[1])+1])

        t = (source_phrase, target_phrase)

        if t in dictonary:
            dictonary[t] += 1
        else:
            dictonary[t] = 1

    return dictonary

def combineDictionaries(phrase_pair_dict_all, phrase_pair_dict_n):

    for phrase_pair, count in phrase_pair_dict_n.iteritems():
        if phrase_pair in phrase_pair_dict_all:
            phrase_pair_dict_all[phrase_pair] += count
        else:
            phrase_pair_dict_all[phrase_pair] = count

    return phrase_pair_dict_all

def get_base_total_score(target_sentence):
    return float(target_sentence.split("|||")[3].strip())

def get_n_best_list_sentence_index(target_sentence):
    return int(target_sentence.split("|||")[0].strip())

# def get_sentence(target_sentence):
#     return "".join(target_sentence.split("|||")[1].split("|")[0:-1:2]).strip().replace("  ", " ").lower()

def get_phrase_pair_lists_and_dicts(source_sentence, n_best_list):
    print "Getting phrase pairs"

    # source_sentence - source sentence (not parsed)
    # n_best_list - 100 best translations of source sentence (not parsed)

    phrase_pair_dict_n_list = []
    phrase_pair_dict_all = {}

    for target_sentence in n_best_list:

        phrase_pair_dict_n = {}

        source_sentence_words = get_source_sentence_words(source_sentence)
        alignments = target_sentence.split("|||")[1].split("|")[1:-1:2]
        target_phrases = target_sentence.split("|||")[1].split("|")[0:-1:2]

        for target_phrase, alignment in zip(target_phrases, alignments):
            # print "Getting alignments and phrase pairs for " + target_sentence 
            alignment = alignment.split("-")
            source_phrase = " ".join(source_sentence_words[int(alignment[0]):int(alignment[1])+1])

            t = (source_phrase, target_phrase)

            if t in phrase_pair_dict_n:
                phrase_pair_dict_n[t] += 1
            else:
                phrase_pair_dict_n[t] = 1

            if t in phrase_pair_dict_all:
                phrase_pair_dict_all[t] += 1
            else:
                phrase_pair_dict_all[t] = 1            

        phrase_pair_dict_n_list.append(phrase_pair_dict_n)

    return phrase_pair_dict_n_list, phrase_pair_dict_all

# n best list, sbleu score list and total base score list

n_best_list_file_name = "test.output.2.nbest2"
sblue_score_list_file_name = "sbleu.txt"

class NBestList(object):
    def __iter__(self):
        for line in open(n_best_list_file_name):
            yield line.strip().lower()

def get_n_best_list_sblue_score_list_and_total_base_score_list(source_sentence_index, start_line_n_best_list_list):
    print "Getting n_best_list"

    start_line_index = start_line_n_best_list_list[source_sentence_index]
    stop_line_index = start_line_n_best_list_list[source_sentence_index+1]

    n_best_list = []
    sblue_score_list = []
    total_base_score_list = []

    for line_index in xrange(start_line_index,stop_line_index):
        target_sentence = linecache.getline(n_best_list_file_name, line_index).strip().lower()
        # print "Adding to n best list"
        n_best_list.append(target_sentence)

        # print "Getting total base score"
        total_base_score = get_base_total_score(target_sentence)
        total_base_score_list.append(total_base_score)

        # print "Getting sbleu score"
        sbleu_score = float(linecache.getline(sblue_score_list_file_name, line_index).strip())
        sblue_score_list.append(sbleu_score)

    return n_best_list, total_base_score_list, sblue_score_list

def get_start_line_n_best_list_list():

    start_line_n_best_list_list = []
    n_best_list = NBestList()
    last_index = -1

    for i, target_sentence in enumerate(n_best_list):
        source_sentence_index = get_n_best_list_sentence_index(target_sentence)
        if (source_sentence_index != last_index):
            start_line_n_best_list_list.append(i+1)
            last_index = source_sentence_index

    start_line_n_best_list_list.append(i+1)
    return start_line_n_best_list_list

source_sentence_list_file_name = "test.input.tok.1"

class SourceSentenceList(object):
    def __iter__(self):
        for line in open(source_sentence_list_file_name):
            yield line.strip().lower()

def main():

    source_sentence_list = SourceSentenceList()

    start_line_n_best_list_list = get_start_line_n_best_list_list()

    for i, source_sentence in enumerate(source_sentence_list):
        print "Getting everything for source_sentence #{}".format(i+1)
        n_best_list, sblue_score_list, total_base_score_list = get_n_best_list_sblue_score_list_and_total_base_score_list(i, start_line_n_best_list_list)
        phrase_pair_dict_n_list, phrase_pair_dict_all = get_phrase_pair_lists_and_dicts(source_sentence, n_best_list)





