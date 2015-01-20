[TOC]

Learning Continuous Phrase Repesentations for Translation Modelling
===================================================================

* __Authors:__ Kerry Zhang, Richard Luong
* __Student-ID:__ 2014403073, 2014403075

Description
-----------

Our project is to implement Phrase Embedding, as described in the paper by Gao et al [\[1\]](http://research.microsoft.com/pubs/211749/nn4smt.acl.v9.pdf).

So when the system is trained, it will output a real number vector given a source sentence. For two similar sentence, the output differences should be smaller, than for two completely different ones.

This is done by a neutal network, with one hidden layer. The first layer has the same number of nodes as the vocabulary size of both lanugages. The second and the third layer is set to 100 nodes.

The initialization of the weights of the first layer to the hidden layer, denoted as `W1`, is done by using a bilingual topic distribution, in this case solved by using stochastic gradient descent. The weights between the hidden layer and the third layer, `W2`, is an identity matrix.

Dependencies
------------

* pyhton 2.7+
* [NumPy](http://www.numpy.org/)
* [Gensim](https://radimrehurek.com/gensim/index.html)

How to run
----------

Unzip the archive.

Before the first run, `W1` needs to be initialized, using [Latent Dirchlet Allocation](#latent-dirchlet-allocation).

	python preprocessing.py $corpus_reference $corpus_target $n_best_list $reference_file

where `$corpus_reference` and `$corpus_target` are paths to the parallell corpuses the bilingual topic modelling should be based on, `$n_best_list` is a path to a list of the n best translation of a source sentence and `$reference_file` is a path to the reference translation of those source sentences.

After that the system should be trained with

	python main.py $source_file_name $n_best_list

where `$source` is a path to the source sentences and `$n_best_list` is a path to the n best translation of these sentences.

Results
-------

When the system is trained, the weights of the neural network are written to the files `W1.txt` and `W2.txt`.

