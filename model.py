import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

def model_processing(X_train_pp, y_train, X_validation, y_validation):
    def LeNet(x):    
        # Hyperparameters
        padding = 'VALID'
        # TODO: Layer 1: Convolutional. Input = 32x32xinput_channels. Output = 28x28x6.
        conv1 = tf.nn.conv2d(x, weights['layer_1'], strides=[1,1,1,1], padding='VALID') + biases['layer_1']
        # TODO: Activation.
        conv1 = tf.nn.relu(conv1)
        # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
        conv1 = tf.nn.avg_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        # TODO: Layer 2: Convolutional. Output = 10x10x16.
        conv2 = tf.nn.conv2d(conv1, weights['layer_2'], strides=[1,1,1,1], padding='VALID') + biases['layer_2']
        # TODO: Activation.
        conv2 = tf.nn.relu(conv2)
        # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = tf.nn.avg_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        # TODO: Flatten. Input = 5x5x16. Output = 400.
        flatten = tf.contrib.layers.flatten(conv2)
        # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
        fc3 = tf.add(tf.matmul(flatten, weights['fully_connected1']),biases['fully_connected1'] )
        # TODO: Activation.
        fc3 = tf.nn.relu(fc3)
        # Dropout
        fc3_drop = tf.nn.dropout(fc3, keep_prob)
        # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
        fc30 = tf.add(tf.matmul(fc3_drop, weights['fully_connected12']),biases['fully_connected12'] )
        # TODO: Activation.
        fc30 = tf.nn.relu(fc30)
        # Dropout
        fc30_drop = tf.nn.dropout(fc30, keep_prob)
        # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
        fc4 = tf.add(tf.matmul(fc30_drop, weights['fully_connected2']), biases['fully_connected2'])
        # TODO: Activation.
        fc4 = tf.nn.relu(fc4)
        # Dropout
        fc4_drop = tf.nn.dropout(fc4, keep_prob)
        # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
        fc5 = tf.add(tf.matmul(fc4_drop, weights['fully_connected3']), biases['fully_connected3'])
        logits = fc5
        return logits
    EPOCHS = 10
    BATCH_SIZE = 128
    input_channels = len(X_train_pp[0][0][0])
    mu = 0
    sigma = 0.1
    # Training Pipeline
    rate = 0.001
    weights = {
        'layer_1': tf.Variable(tf.truncated_normal([5, 5, input_channels, 6])),
        'layer_2': tf.Variable(tf.truncated_normal([5, 5, 6, 16])),
        'fully_connected1': tf.Variable(tf.truncated_normal(shape=(400, 250), mean = mu, stddev = sigma)),
        'fully_connected12': tf.Variable(tf.truncated_normal(shape=(250, 120), mean = mu, stddev = sigma)),
        'fully_connected2': tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma)),
        'fully_connected3': tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    }

    biases = {
        'layer_1': tf.Variable(tf.zeros(6)),
        'layer_2': tf.Variable(tf.zeros(16)),
        'fully_connected1': tf.Variable(tf.zeros(250)),
        'fully_connected12': tf.Variable(tf.zeros(120)),
        'fully_connected2': tf.Variable(tf.zeros(84)),
        'fully_connected3': tf.Variable(tf.zeros(43))
    }
    #Features and labels
    x = tf.placeholder(tf.float32, (None, 32, 32, input_channels))
    y = tf.placeholder(tf.int32, (None))
    keep_prob = tf.placeholder(tf.float32)
    one_hot_y = tf.one_hot(y, 43)

    logits = LeNet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)

    # Model Evaluation
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train_pp)
        
        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train_pp, y_train = shuffle(X_train_pp, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train_pp[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.7})
                
            validation_accuracy = evaluate(X_validation, y_validation)
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()
            
        saver.save(sess, 'lenet')
        print("Model saved")


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples