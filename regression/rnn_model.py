import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell


def gen_RNN(_X, _istate, _weights, _biases, n_input, n_hidden, n_steps):

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate)

    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']


class RNN(object):
    def __init__(self, n_steps, n_input, n_hidden, n_classes):
        self.x = tf.placeholder("float", [None, n_steps, n_input])
        # Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
        self.istate = tf.placeholder("float", [None, 2*n_hidden])
        self.y = tf.placeholder("float", [None, n_classes])

        # Define weights
        weights = {
            'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
        }
        biases = {
            'hidden': tf.Variable(tf.random_normal([n_hidden])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        self.pred = gen_RNN(self.x, self.istate, weights, biases,
                       n_input, n_hidden, n_steps)

        # Define loss and optimizer
        self.cost = tf.reduce_sum(tf.pow((self.pred - self.y), 2), 0) # Softmax loss
        # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.y)) # Softmax loss
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

        # Evaluate model
        # correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(self.y,1))
        base_correct = tf.zeros(tf.shape(self.y), tf.int32)
        self.correct_pred = tf.equal(tf.cast((tf.abs(self.pred - self.y) * 2), tf.int32), base_correct)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"))
