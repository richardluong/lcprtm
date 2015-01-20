[TOC]

Learning Continuous Phrase Repesentations for Translation Modelling
===================================================================

* __Authors:__ Kerry Zhang, Richard Luong
* __Student-ID:__ 2014403073, 2014403075

__Distribution of work between the authors__: To be sure that both of us made equal amount of work, the [pair programming](http://en.wikipedia.org/wiki/Pair_programming) technique was applied. 

Description
-----------

Our project is to implement Phrase Embedding, as described in the paper by Gao et al [\[1\]](http://research.microsoft.com/pubs/211749/nn4smt.acl.v9.pdf).

So when the system is trained, it will output a real number vector given a source sentence. For two similar sentence, the output differences should be smaller, than for two completely different ones.

This is done by a neural network, with one hidden layer. The first layer has the same number of nodes as the vocabulary size of both lanugages. The second and the third layer is set to 100 nodes.

The initialization of the weights of the first layer to the hidden layer, denoted as `W1`, is done by using a bilingual topic distribution, in this case solved by using Latent Dirchlet Allocation. The weights between the hidden layer and the third layer, `W2`, is an identity matrix.

Optimization of the weights are done with [Stochastic gradient descent](http://en.wikipedia.org/wiki/Stochastic_gradient_descent). The training is stopped according to early stop principle, as suggested in the paper.

Dependencies
------------

* python 2.7+
* [NumPy](http://www.numpy.org/)
* [Gensim](https://radimrehurek.com/gensim/index.html)

How to run
----------

Unzip the archive.

Before the first run, `W1` needs to be initialized, using [Latent Dirchlet Allocation](http://en.wikipedia.org/wiki/Latent_Dirichlet_allocation).

	python preprocessing.py $corpus_source $corpus_target $n_best_list $reference_file

where `$corpus_source` and `$corpus_target` are paths to the parallell corpuses the bilingual topic modelling should be based on, `$n_best_list` is a path to a list of the n best translation of source sentences and `$reference_file` is a path to the reference translation of those source sentences. Sentence-level BLEU score is also computed during this stage. The W1 initialization and BLEU score can be found in `data/weight_initialization.gz` and `sbleu.txt`

After that the system should be trained with

	python main.py $source_sentence_file $n_best_list

where `$source` is a path to the source sentences and `$n_best_list` is a path to the n best translation of these sentences.

Results
-------

When the system is trained, the weights of the neural network are written to the files `W1.gz` and `W2.gz`.

With the given corpus and n best list from our teacher, an instance of our program managed to increase the expected BLEU score from 1 to 10.

Discussion
----------

Due to limited computing power, we were not able to run the program until it converges. A small training set (3 sentence pairs, 1 sentence test set) converged after 3 hours.

References
----------

[1]: __Learning Continuous Phrase Representations for Translation Modeling__ by Gao et. al. 2013. 

