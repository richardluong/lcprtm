from __future__ import division
import numpy as np
from gensim import corpora


class CPTMNeuralNetwork():

    def __init__(self, sizes, weights=None):
        self.num_layers = len(sizes)
        self.sizes = sizes
        if weights:
            self.weights = weights
        else:
            weights = []
            # w1 is dx100 matrix where each column sums to 1
            w1 = np.random.dirichlet(np.ones(sizes[0]), size=sizes[1]).transpose()
            w2 = np.identity(sizes[1])
            weights.append(w1)
            weights.append(w2)
            self.weights = np.array(weights)

    def get_z(self, weights, a):
        """
        z is the input vector to a layer.
        E.g. for layer 1 (the first hidden layer) z = W_1.tranpose * x
        where x is the input layer
        """
        return np.dot(weights.transpose(), a)

    def get_y(self, z):
        return np.vectorize(np.tanh)(z)

    def get_theta_gradients(self, x_f, x_e):
        """
        @x_f : bag of words of foreign phrase
        @x_e : bag of words of English phrase
        """
        W1 = self.weights[0]
        W2 = self.weights[1]

        z1_f = self.get_z(W1, x_f)
        y1_f = self.get_y(z1_f)
        z2_f = self.get_z(W2, y1_f)
        y2_f = self.get_y(z2_f)

        z1_e = self.get_z(W1, x_e)
        y1_e = self.get_y(z1_e)
        z2_e = self.get_z(W2, y1_e)
        y2_e = self.get_y(z2_e)

        W2_gradient = np.dot(y1_f, (y2_e * tanh_prime(z2_f)).transpose())\
            + np.dot(y1_e, (y2_f * tanh_prime(z2_e)).transpose())

        tmp1 = np.dot(W2, y2_e * tanh_prime(z2_f)) * tanh_prime(z1_f)
        tmp2 = np.dot(W2, y2_f * tanh_prime(z2_e)) * tanh_prime(z1_e)
        W1_gradient = np.dot(x_f, tmp1.transpose()) + np.dot(x_e, tmp2.transpose())

        return W1_gradient, W2_gradient

    def feedforward(self, x):
        """
        Returns the output vector of the network, if @x is the input
        """
        for w in self.weights:
            x = np.vectorize(np.tanh)(np.dot(w.transpose(), x))
        return x

    def update_mini_batch(self, mini_batch, eta, dictionary, error_term_dict):
        """
        Update the network's weights (W_1, W_2).
        Calculates the gradients (Equation 7 in the paper)
            sum_(f, e) {dL/dtheta = error_term * theta gradients}

        @mini_batch : dictionary of all phrase pairs. mini_batch[(f, e)] = c
                            where f = source phrase, e = target phrase, c = count (# times observed)
                            e.g. dictionary[("vamos", "let's go")] = 3
                      observed in a source sentence F_i and its corresponding
                      N-best list GEN(F_i)
        @eta : float, learnig factor
        @dictionary : class gensim.corpora.Dictionary
        @error_term_dict : dict in form of: error_term_dict[(f, e)] = error_term(f, e)
        """

        # Gradients for W1 and W2
        d_W1 = np.zeros((self.sizes[0], self.sizes[1]))
        d_W2 = np.zeros((self.sizes[1], self.sizes[2]))

        # Equation 7 in the paper.
        # For each sentence pair (f, e) observed between source sentence F_i
        # and its corresponding N-best list GEN(F_i)
        #   calculate error_f_e and the gradient it contributes to
        for f, e in mini_batch.keys():

            x_f = np.zeros((self.sizes[0], 1))
            for word in f.split(" "):
                word_index = dictionary.token2id[word]
                x_f[word_index] += 1
            x_e = np.zeros((self.sizes[0], 1))
            for word in e.split(" "):
                word_index = dictionary.token2id[word]
                x_e[word_index] += 1

            W1_gradient, W2_gradient = self.get_theta_gradients(x_f, x_e)
            count = mini_batch[(f, e)]
            error_term = error_term_dict[(f, e)]

            d_W1 += count * error_term * W1_gradient
            d_W2 += count * error_term * W2_gradient

        # gradient descend
        self.weights[0] -= eta * d_W1
        self.weights[1] -= eta * d_W2


def tanh_prime(x):
    return np.vectorize(lambda x: 1 - (np.tan(x)**2))(x)


def phrase_to_bow_vector(phrase, dictionary):
    """
    Returns the bag of word vector representation of a phrase.
    @phrase : string, where each word is separated by a white space
    @dictionary : gensim.corpora.Dictionary
    """
    x = np.zeros((len(dictionary), 1))
    for word in phrase.split(" "):
        word_index = dictionary.token2id[word]
        x[word_index] += 1
    return x


def get_error_term_dict(
        phrase_pair_dict_all, phrase_pair_list_hypothesis,
        sBleu_list, xBleu,
        Ej_translation_probability_list,
        new_feature_weight=1):
    """
    From Equation 14
    Returns the error term for each phrase pair observed
    between F_i and all sentences in GEN(F_i)

    @phrase_pair_dict_all : dictionary of all phrase pairs. mini_batch[(f, e)] = c
                where f = source phrase, e = target phrase, c = count (# times observed)
                e.g. dictionary[("vamos", "let's go")] = 3
                observed in a source sentence F_i and all sentences in its corresponding
                N-best list GEN(F_i)
    @sBleu_list : list of sBleu values for all sentences in GEN(F_i)
    @xBleu
    @Ej_translation_probability_list :
        list of translation probabilitiesfor all sentences in GEN(F_i)
    @phrase_pair_list_hypothesis : list of dicts, where each dict
        is the phrase pair dict between F_i and a hypothesis translation in GEN(F_i)
    """
    # See equation 14
    U_list = map(lambda sBleu: sBleu - xBleu, sBleu_list)

    error_dict = {}
    for f, e in phrase_pair_dict_all.keys():
        for j, phrase_pair_dict_j in enumerate(phrase_pair_list_hypothesis):
            U_j = U_list[j]
            P_j = Ej_translation_probability_list[j]
            N_j = phrase_pair_dict_j[(f, e)]
            error_dict[(f, e)] = U_j * P_j * new_feature_weight * N_j
    return error_dict


def new_feature_value(nn, phrase_pair_dict, dictionary):
    """
    Calculates the new feature value of a translation,
    using Equation 4 in the paper.

    @nn : class CPTMNeuralNetwork, our neural network
    @phrase_pair_dict : dictionary of all phrase pairs. phrase_pair_dict[(f, e)] = c
                            where f = source phrase, e = target phrase, c = count (# times observed)
                            e.g. dictionary[("vamos", "let's go")] = 3
                        observed between a source sentence F_i
                        and the translation whose feature value we want to calculate
    @dictionary : class gensim.corpora.Dictionary
    """
    feature_value = 0
    for f, e in phrase_pair_dict.keys():
        x_f = phrase_to_bow_vector(f, dictionary)
        x_e = phrase_to_bow_vector(e, dictionary)

        y_f = nn.feedforward(x_f)
        y_e = nn.feedforward(x_e)

        count = phrase_pair_dict[(f, e)]
        feature_value += count * np.dot(y_f.transpose(), y_e)
    return feature_value


input_dim = 10
hidden_dim = 5
output_dim = 5

nn = CPTMNeuralNetwork([input_dim, hidden_dim, output_dim])
x_f = np.ones((input_dim, 1))
x_e = np.ones((input_dim, 1))
W1_gradient, W2_gradient = nn.get_theta_gradients(x_f, x_e)
print "W1_gradient", W1_gradient.shape, "W2_gradient", W2_gradient.shape

dictionary_path = '../my_files/deerwester.dict'
dictionary = corpora.Dictionary.load(dictionary_path)
phrase_pair_dict = {
    ("fawn nunnery", "woods spiders"): 3
}


"""
def get_error_term_dict(
        phrase_pair_dict_all, phrase_pair_list_hypothesis
        sBleu_list, xBleu,
        Ej_translation_probability_list,
        new_feature_weight=1):
"""
"""
phrase_pair_dict_all = {
    ("fawn nunnery", "woods spiders"): 3
}
phrase_pair_list_hypothesis = [
    {("fawn nunnery", "woods spiders"): 3}
]
sBleu_list = [4]
xBleu = 5
Ej_translation_probability_list = [1]
print get_error_term_dict(
    phrase_pair_dict, phrase_pair_list_hypothesis,
    sBleu_list, xBleu,
    Ej_translation_probability_list)
"""

"""
print
print "Weights before update mini batch"
print nn.weights
nn.update_mini_batch(phrase_pair_dict, 0.001, dictionary)
print
print "Weights after update mini batch"
print nn.weights

phrase1 = "fawn nunnery"
x1 = np.zeros((input_dim, 1))
for word in phrase1.split(" "):
    word_index = dictionary.token2id[word]
    x1[word_index] += 1
print
print "Phrase:", phrase1
print "Word vector:"
print x1
print "Network output (feedforward)"
print nn.feedforward(x1)
"""

# w1 = nn.weights[0]
# z_1 = nn.get_z(w1, np.ones((10,1)))
# y_1 = nn.get_y(z_1)
# print y_1
