def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    import tensorflow as tf

    tf.set_random_seed(1)  # so that your "random" numbers match ours


    W1 = tf.get_variable("W1", [500, 784], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [500, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [250, 500], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [250, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [125, 250], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [125, 1], initializer=tf.zeros_initializer())
    W4 = tf.get_variable("W4", [25, 125], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b4 = tf.get_variable("b4", [25, 1], initializer=tf.zeros_initializer())
    W5 = tf.get_variable("W5", [10, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b5 = tf.get_variable("b5", [10, 1], initializer=tf.zeros_initializer())


    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  }

    return parameters