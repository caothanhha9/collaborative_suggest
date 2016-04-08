import data_helpers
from rnn_model import RNN
import tensorflow as tf
import numpy as np
import os


class Train(object):
    def __init__(self, learning_rate, n_hidden, batch_size, training_iters,
                 out_dir, checkpoint_step, display_step=10):
        self.learning_rate = learning_rate
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.training_iters = training_iters
        self.out_dir = out_dir
        self.checkpoint_step = checkpoint_step
        self.display_step = display_step

    def run(self):
        input_data, label_data = data_helpers.load_data()
        n_steps = 3
        n_input = 1
        n_classes = 1
        n_hidden = self.n_hidden
        batch_size = self.batch_size
        training_iters = self.training_iters
        display_step = self.display_step
        checkpoint_step = self.checkpoint_step
        # batches = data_helpers.batch_gen(zip(input_data, label_data), 2)
        # for batch in batches:
        #     x_batch, y_batch = zip(*batch)
        #     print('-' * 50)
        #     print(x_batch)
        #     print(y_batch)

        new_rnn = RNN(n_steps=n_steps, n_input=n_input,
                      n_hidden=self.n_hidden, n_classes=n_classes)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(new_rnn.cost) # Adam Optimizer
        # global_step = tf.Variable(0, name="global_step", trainable=False)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # grads_and_vars = optimizer.compute_gradients(new_rnn.cost)
        # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.join(self.out_dir, "checkpoints")
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            # Keep training until reach max iterations
            while step * batch_size < training_iters:
                # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                batch_xs, batch_ys = data_helpers.get_random_batch(input_data, label_data, batch_size)
                # Reshape data to get 28 seq of 28 elements
                batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
                # Fit training using batch data
                sess.run(optimizer, feed_dict={new_rnn.x: batch_xs, new_rnn.y: batch_ys,
                                               new_rnn.istate: np.zeros((batch_size, 2*n_hidden))})
                if step % display_step == 0:
                    # Calculate batch accuracy
                    acc = sess.run(new_rnn.accuracy, feed_dict={new_rnn.x: batch_xs, new_rnn.y: batch_ys,
                                                        new_rnn.istate: np.zeros((batch_size, 2*n_hidden))})
                    # Calculate batch loss
                    loss = sess.run(new_rnn.cost, feed_dict={new_rnn.x: batch_xs, new_rnn.y: batch_ys,
                                                     new_rnn.istate: np.zeros((batch_size, 2*n_hidden))})
                    print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                          ", Training Accuracy= " + "{:.5f}".format(acc)
                if step % checkpoint_step == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))
                step += 1
            print "Optimization Finished!"

new_train = Train(learning_rate=0.001, n_hidden=12, batch_size=2, training_iters=10000, out_dir='models',
                  checkpoint_step=1000)
new_train.run()
