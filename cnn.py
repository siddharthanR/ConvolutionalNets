from datetime import datetime as dt
import input_data as idt
import math as mt
import matplotlib.pyplot as plot
import tensorflow as tf
import numpy as np

#load mnist data
mnist = idt.read_data_sets('data/', one_hot = True)

#hyperparameters
learning_rate = 0.0001
batch_size = 100
step = 10

#input and output nuerons
ip_nuerons = 128
op_nuerons = 10

#moving of pixel position
#strides : number of pixel to move
s = 1

#filter for convolutional blocks
filter1 = 5
num_filter1 = 16
filter2 = 5
num_filter2 = 36

#grey scale
channel = 1 

#define image width and total size
img_size = 28
image_flat_size = 784

#graph input
x = tf.placeholder(tf.float32 ,[None, image_flat_size])
y = tf.placeholder(tf.float32, [None, op_nuerons])


#define weight matrix
def weight_matrix(shape):
	return tf.Variable(tf.random_normal(shape))


#define baise matrix
def bias_matrix(shape):
	return tf.Variable(tf.random_normal(shape))


#single convolutional block 
#( convolution + relu )
def convolutional_block(x, weights, bias):
	conv = tf.nn.conv2d(x, weights, strides = [1, s, s, 1],
				padding = 'SAME')
	conv = tf.nn.bias_add(conv, bias)
	return tf.nn.relu(conv)


#down sampling (maxpooling)
def down_sampling(x, k, s):
	return tf.nn.max_pool(x, ksize = [1, k, k, 1],
				strides = [1, s, s, 1] ,padding = 'SAME')


#fully connected network
def convolutional_nets(x, weights, bias):
	#convolutional block (convolution + relu + pooling)
	conv = convolutional_block(x, weights, bias)
	conv = down_sampling(conv, 2, 2)
	return conv
	
#flattening of convolutional output
def layer_flatten(layer):
	layer_shape = layer.get_shape()
	features = layer_shape[1:4].num_elements()
	layer_flat = tf.reshape(layer, [-1, features])
	return layer_flat, features


#new layer
def new_layer(x, weights, bias):

	#single layer (input * weight) + bias
	l = tf.add(tf.matmul(x, weights), bias)
	return tf.nn.relu(l)


#optimizing network 
def optimize_network(epoch):
	error = [ ]
	acc = [ ]
	batch_cost = 0
	batch_accuracy = 0
	#run the tensorflow session
	with tf.Session() as session:

		#initialize all the variables
		session.run(tf.global_variables_initializer())
		
		#time we started optimizing
		starting_time = dt.now().time()
		print("\nstarting time:{0}".format(starting_time))
		for i in range(epoch):

			#splitting of data into several batches 
			batch_x, batch_y = mnist.train.next_batch(batch_size)

			#data to be fed into the network
			data = {
				x : batch_x,
				y : batch_y
					}

			#get the accuracy and loss
			k = session.run([op, accuracy, loss], feed_dict = data)

			#calculate batchwise cost and accuracy
			batch_cost += round((k[2] / batch_size), 0)
			batch_accuracy += round((k[1] / batch_size), 0)
			
			#append cost and accuracy onto list for each batches
			error.append(batch_cost)
			acc.append(batch_accuracy)

			if i % 10 == 0 :
				msg = "epoch:{0}  accuracy:{1}".format(i, batch_accuracy)
				print(msg," cost = {0}".format(batch_cost))

		#calculating ending time
		#time it completes the optimization 
		ending_time = dt.now().time()
		print("\nending time:{0}".format(ending_time))
		h = ending_time.hour - starting_time.hour
		m = ending_time.minute - starting_time.minute
		s = ending_time.second - starting_time.second
		print("total time taken: hours:{0} minutes:{1} seconds:{2}".format(h, m, s))
		print(len(error), " ",epoch)
		plot_training_error(error, epoch)


#plot images
def plot_image(image, img_size):
	plot.imshow(image.reshape((img_size, img_size)))
	plot.show()


#plot images and its class
def plot_image_class(image, label, pred = None):
	assert len(image) == len(label) == 9

	#create a subplot 
	fig, axes = plot.subplots(3, 3)
	fig.subplots_adjust(hspace = 0.3, wspace = 0.3)
	
	#visualize the input images and its respective class
	for i, ax in enumerate(axes.flat):
		#reshape the image to 784 (28 * 28)
		ax.imshow(image[i].reshape((img_size, img_size)))

		#set the x labels
		if pred is None:
			xlabel = "True: {0}".format(label[i])
		else:
			xlabel = "True: {0} pred: {1}".format(label[i], pred[i])

		#set x label
		ax.set_xlabel(xlabel)

		#remove the ticks from plot
		ax.set_xticks([])
		ax.set_yticks([])

		#show the image
	plot.show()

#plotting filters of different layers
def plot_filters(weights, input_channel = 0):

	#create the session to get the weights
	with tf.Session() as session:
		session.run(tf.global_variables_initializer())
		values = session.run(weights)
	
	#get the number of filters
	filters = values.shape[3]

	#get the minimum and maximum values
	min_value = np.min(values) 
	max_value = np.max(values)

	#compute number of grids
	num_grid = mt.ceil(mt.sqrt(filters))

	#create a subplot of size grid * grid (rows = grid, columns = grid)
	fig, axes = plot.subplots(num_grid, num_grid)

	#iterate through all the filters
	for i, ax in enumerate(axes.flat):
		if i < filters:

			#weights are 4d tensor
			#1 width of the filter
			#2 height of the filter
			#3 input channel
			#4 number of filters
			#we need input channel and number of filters
			img = values[:, :, input_channel, i]
			ax.imshow(img, vmin = min_value, vmax = max_value, cmap = 'binary')
		
		#remove the ticks from the plot
		ax.set_xticks([])
		ax.set_yticks([])
	
	#show the image
	plot.show()


#plot different convolutional layers
def plot_layers(layer, image):

	#define the input data to the network
	data = {
		x : image.reshape([1, image_flat_size]) 
			}

	#create a session to get the convolutional of an image
	with tf.Session() as session:
		session.run(tf.global_variables_initializer())
		values = session.run(layer, feed_dict = data)

	#shape = (image number, width, height, filters) 
	#get the number of filters 
	filters = values.shape[3]

	#number of grids to use
	num_grid = mt.ceil(mt.sqrt(filters))

	#create a subplot of grid size (row = grid, column = grid)
	fig, axes = plot.subplots(num_grid, num_grid)

	#iterate through all the filters
	for i, ax in enumerate(axes.flat):
		if i < filters:
			#convolution output is a 4d-tensor
			#1 image number 
			#2 image width
			#3 image height
			#4 number of filters (defines number of outputs)
			#we need filters and image number.
			img = values[0, :, :, i]
			ax.imshow(img, cmap = 'binary')

		#remove ticks from the plot
		ax.set_xticks([])
		ax.set_yticks([])

	#show the image
	plot.show()



#convolution operation 2 blocks
#first convolution block
#define weights and biases
x_conv = tf.reshape(x, shape=[-1, img_size, img_size, channel])

weights1 = weight_matrix([filter1, filter1, channel, num_filter1])
bias1 = bias_matrix([num_filter1])
conv1 = convolutional_nets(x_conv, weights1, bias1)


#second convolution block
#define weights and biases
weights2 = weight_matrix([filter2, filter2, num_filter1, num_filter2])
bias2 = bias_matrix([num_filter2])
conv2 = convolutional_nets(conv1, weights2, bias2)


#flattening of second convolutional output 
#second layer outputs 4D tensor 
#we need 2d tensor
flat_layer, features = layer_flatten(conv2)


#fully connected network layer 1
#weights and biases
weights3 = weight_matrix([features, ip_nuerons])
bias3 = bias_matrix([ip_nuerons])
l1 = new_layer(flat_layer, weights3, bias3)


#fully connected network layer 2 (output layer)
#weights and biases
weights4 = weight_matrix([ip_nuerons, op_nuerons])
bias4 = bias_matrix([op_nuerons])
out = new_layer(l1, weights4, bias4)


#apply softmax function to the output
prediction = tf.nn.softmax(out)

#cost function 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
		logits = out, labels = y))

#adam optimizer
op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

#true label and predicted label interms of o's and 1's
true_class = tf.argmax(y, axis = 1)
pred_class = tf.argmax(prediction, axis = 1)

#prediction accuracy
correct_prediction = tf.equal(pred_class, true_class)

#tf.reduc_mean() - calculates the average
#tf.cast() - typecasting boolean to float 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimize_network(50)