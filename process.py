from sklearn.model_selection import train_test_split
import os
from image_to_input import get_input_data
import pandas as pd
import numpy as np
import random 
import tensorflow as tf

yield_csv = pd.read_csv('data_for_image_download.csv',header=None)
output = []
hhid_list = []
for hhid, yld, lat, lon in yield_csv.values:
	output.append(yld)
	hhid_list.append(str(hhid)[:-2])

hhid_input = {}
hhid_output = {}
counter = 0
for file in os.listdir("image_data"):
	if file.endswith(".tif"):
		input_array = get_input_data("image_data/" + file)
		hhid_input[counter] = input_array
		hhid_output[counter] = output[counter]
		counter += 1

training_set_X = []
training_set_Y = []
testing_set_X = []
testing_set_Y = []
n_train = int(counter * 0.9)

lst = list(range(counter))
for i in range(n_train):
	r_key = random.choice(lst)
	training_set_Y.append(hhid_output[r_key])
	training_set_X.append(hhid_input[r_key])
	lst.remove(r_key)

for k in lst:
	testing_set_Y.append(hhid_output[k])
	testing_set_X.append(hhid_input[k])

learning_rate = 0.01

l_train = len(training_set_Y)
l_test = len(testing_set_Y)

num_input = 350
n_hidden_1 = 64
n_hidden_2 = 64

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, 1])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, 1]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([1]))
}

# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
prediction = neural_net(X)
loss_op = tf.reduce_mean(tf.sqrt(tf.losses.mean_squared_error(Y, prediction)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
accuracy = tf.sqrt(tf.losses.mean_squared_error(Y, prediction))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    display_step = 10
    lst = list(range(n_train))
    for step in range(0, 10000):
        sample = random.sample(lst, 20)
        batch_x = []
        batch_y = []
        for i in sample:
            batch_x.append(training_set_X[i])
            batch_y.append(training_set_Y[i])
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y).reshape((20,1))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    
    for step in range(0, 10):
        sess.run(train_op, feed_dict={X: np.array(training_set_X), Y: np.array(training_set_Y).reshape((l_train, 1))})
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: np.array(training_set_X), Y: np.array(training_set_Y).reshape((l_train, 1))})
        print("Final Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

    print "Optimization Finished!"


    print "Testing Yields: ", testing_set_Y
    print "Predictions: ", prediction.eval(feed_dict={X: np.array(testing_set_X)}, session=sess)
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={X: np.array(testing_set_X), Y: np.array(testing_set_Y).reshape((l_test, 1))})

