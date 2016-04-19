import numpy as np


def batch_iter(data, batch_size, num_epochs):
    """
    :param data
    :param batch_size
    :param num_epochs
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    if data_size % batch_size == 0:
        num_batches_per_epoch = data_size / batch_size
    else:
        num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def batch_gen(data, batch_size):
    data = np.array(data)
    data_size = len(data)
    if data_size % batch_size == 0:
        num_batches = data_size / batch_size
    else:
        num_batches = int(len(data)/batch_size) + 1
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]


def get_random_batch(input_data, target, _batch_size):
    input_data_np = np.array(input_data)
    target_np = np.array(target)
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(input_data)))
    batch_indices = shuffle_indices[:_batch_size]
    batch_input = input_data_np[batch_indices]
    batch_target = target_np[batch_indices]
    # batch_input = []
    # batch_target = []
    # for _index in batch_indices:
    #     batch_input.append(input_data[_index])
    #     batch_target.append(target[_index])
    return [batch_input, batch_target]


def load_data():
    input_data = [[5.0, 4.0, 5.0], [5.0, 5.0, 4.0], [5.0, 5.0, 5.0], [2.0, 1.0, 2.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]
    target = [[5.0], [4.0], [5.0], [2.0], [1.0], [1.0]]

    input_data_np = np.array(input_data)
    target_np = np.array(target)
    return [input_data_np, target_np]


def load_test_data():
    test_data = [[5.0, 4.0, 5.0], [1.0, 1.0, 2.0]]
    test_label = [[5.0], [1.0]]
    test_data_np = np.array(test_data)
    test_label_np = np.array(test_label)
    return [test_data_np, test_label_np]


def load_predict_data():
    test_data = [[5.0, 4.0, 5.0], [1.0, 1.0, 2.0], [2.0, 2.0, 1.0]]
    test_data_np = np.array(test_data)
    return test_data_np
