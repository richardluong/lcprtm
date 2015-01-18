
n_best_list_file_path = "" ;

class MyNBestList(object):
    def __iter__(self):
        for line in open(n_best_list_file_name):
            yield line.strip().lower()

def get_sentence(s):
    return "".join(s.split("|||")[1].split("|")[0:-1:2]).strip().replace("  ", " ").lower()

def get_n_best_list():



