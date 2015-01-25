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

Optimization of the weights are done with [Stochastic gradient descent](http://en.wikipedia.org/wiki/Stochastic_gradient_descent). The training should be stopped according to early stop principle, as suggested in the paper, however we haven't fine tuned the system sufficiently to determine a stop condition as of yet.

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

Output
-------

When the system converges, the weights of the neural network are written to the files `W1.gz` and `W2.gz`.


Experiment Results and Discussion
----------

After multiple overnight test runs of the system that resulted in calculations that outputted NaN due to calculations that accumulated into too large or small numbers, and other issues such as too big gradients resulting in a very unstable gradient descent, we finally managed to get a relatively stable system. The two main parameters that required tuning were: `learning_rate` and `smoothing_factor`, which can be altered at the top of `main.py`. See Equation 6 in the paper by Gao et. al. 2013 for an explanation of the smoothing factor.

As of now, the most stable system we have produced came when the parameters were set as follows: `smoothing_factor = 10` and `learning_rate = 1000`. With these parameter values, we conducted two overnight test runs. 

The first was to test the theoretical validity of our system, i.e. whether or not the implemented calculations of the gradients, as suggested by the paper, actually result in a lowering of the loss function defined as $-xBlue$. To do this, we used 30 training samples (sentence number 200 to 230 in the file `test.input.tok.1` provided by our teacher), performed the gradient descent training on the entire training set and then tested the new average $-xBlue$ score for the entire training set. I.e. we used the training set as our test set in order to test the theoretical validity of our neural network. The results (See Figure `lcprtm/Result/csv/Theoretical_validity_graph.png` ****INSERT GRAPH SOMEHOW???????????????*****) seem to suggest that the system is indeed performing a correct gradient descent.

The second overnight experiment was conducted on a larger training set with 180 training samples and 20 test samples, this time using source sentences 1 to 180 from our input file `test.input.tok.1` as training samples and sentences 181 to 200 as test samples. This choice of training samples means that the result are independent of the result in the first test, as none of the training samples overlap. Again, the result of our experiment suggest that the system works (*******INSERT lcprtm/Result/csv/Smoothing10_200-training-samples_graph.png************), however after 14 epochs the gradients became very large and the system calculations outputted NaN again. Detailed system output can be found in the `lcprtm/Result` directory and the results discussed in this section can be found in the `lcprtm/Result/csv/` directory.

********
TO DO:
lcprtm/Result/csv innheåller de två sista overnight körningarna jag gjorde som typ confirm att systemet funkar någorlunda bra. Bifoga graferna på nåt sätt?
********

Conclusion
----------
We believe that our neural network correctly learns how to minimize the loss function $-xBleu$ and is relatively stable. Increasing `smoothing_factor` will result in smaller gradients and slower training. Because we are getting NaN values (too big or too small numbers) after certain amounts of epochs with large training sets, we believe that `smoothing_factor` should be increased for a more stable system. By how much can only be determined by trial and error, more experiments need to be conducted in order to tune the parameter `smoothing_factor` in particular, and possibly `learning_rate`. 

As of now, the training will run forever as we have not been able to tune the parameters for a sufficiently stable system in order to determine a good convergence criteria.

References
----------

\[1\]: __Learning Continuous Phrase Representations for Translation Modeling__ by Gao et. al. 2013. 

