import tensorflow as tf
import numpy as np
import cv2
import os

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#a simple generative adversarial network for the MNIST dataset
gen_seed_size = 3
minibatch_size = 128

#play with this value more, potentially lower it to fifty
k_disc_train_steps = 1

#run two different examples through the discriminate network using the same weights and biases

#problem with this network: many examples are generated near class boundaries, you need a way to generate more discriminative results

#use experience replay, replaying experiences to the discriminator and generator so that they don't get trapped in local minimums
#check out this paper for solving that issue
#https://arxiv.org/pdf/1711.01575.pdf

#note: the loss for the generator currently is based on both the examples in memory and the currently generated models
#		a solution would be to create a real/dream dataset mix and a real/current_generated dataset mix
#Note: the generator still degrades after about 60,000 steps


#Note: the replay isn't working properly, it's degrading after only 20000 steps
#Note: try updating G once for every k step updates of D, this keeps 

#Note: as of now, the discriminator is learning how to draw zeroes very well
#Note: the generator seems to really just get good at drawing one number at a time, first it was zeroes now it's 3s

#Note: try changing the random data that is fed to the network, right now it's normalized, denormalize it

fake_one_hot = tf.constant([0.0, 0.1], dtype=tf.float32)
real_one_hot = tf.constant([1.0, 0.0], dtype=tf.float32)

class generator():
	def __init__(self, layer_array, noise_size=10):
		#layer_array is an array of tuples that defines the layers of the network
		#
		self.layers = []
		self.noise_size = noise_size

		count = 1
		for i in layer_array:
			layer_vars = self.generate_weights_biases(i, name="layer_" + str(count))
			self.layers.append(layer_vars)
			count += 1

		'''input_weights, input_biases = self.generate_weights_biases([noise_size, 256])
		input_weights'''
	'''def forward_pass(self):
		with tf.name_scope("generator_forward_pass"):'''

	def forward_pass(self, batch_size=128):
		with tf.name_scope("generator_forward_pass"):
			prev_layer_activation = self.seed_noise([batch_size, self.noise_size])
			for i in range(len(self.layers) - 1):
				
				layer_n = self.layers[i]
				mul_value = tf.matmul(prev_layer_activation, layer_n[0])
				add_value = tf.add(mul_value, layer_n[1])
				activation = tf.nn.relu(add_value)
				prev_layer_activation = activation
				print("worked_fine")

			#produce linear output for the last layer
			print("linearizing!")
			op_mul = tf.matmul(prev_layer_activation, self.layers[-1][0])

			
			op_add = tf.add(op_mul, self.layers[-1][1])
		return op_add
	def generate_weights_biases(self, weights_shape, name):
		weights = tf.Variable(tf.random_normal(weights_shape, stddev=0.1), name=name+"_weights")
		biases = tf.Variable(tf.random_normal([weights_shape[1]]), name=name+"_biases")
		return [weights, biases]
	def seed_noise(self, shape):
		return tf.random_normal(shape, stddev=0.1)
		#return np.random.uniform(shape)
	def discriminator_loss(self, discriminator_grade):
		grade_length = tf.shape(discriminator_grade)[0]

		grade_labels = tf.tile(real_one_hot, [grade_length])
		grade_labels = tf.reshape(grade_labels, [grade_length, 2])
		loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_grade, labels=grade_labels)
		return tf.reduce_mean(loss)
	def network_variables(self):
		op_var = []
		for chip in self.layers:
			op_var.extend(chip)
		return op_var

class discriminator():
	def __init__(self, layer_array, input_size=784):
		self.layers = []
		for i in range(len(layer_array) - 1):
			layer_vars = self.generate_weights_biases(layer_array[i], name="layer_" + str(i))
			self.layers.append(layer_vars)
		self.layers.append(self.generate_weights_biases(layer_array[-1], name="output_layer"))

		self.memory_tensor = tf.zeros([1, input_size], dtype=tf.float32)
		print("mem_tens: " + str(self.memory_tensor))
		self.dropout_prob = tf.placeholder(tf.float32)

		self.real_examples = tf.placeholder(tf.float32, [None, input_size], name="discriminator_real_data")
	def generate_weights_biases(self, weights_shape, name):
		weights = tf.Variable(tf.random_normal(weights_shape, stddev=0.1), name=name+"_weights")
		biases = tf.Variable(tf.random_normal([weights_shape[1]]), name=name+"_biases")
		return [weights, biases]
	def forward_pass(self, input_value, batch_size=128):
		#this function will do a forward pass with both the memory tensor and the current generator tensor
		with tf.name_scope("discriminator_forward_pass"):
			prev_layer_activation = input_value
			for i in range(len(self.layers) - 1):
				print("disc_normal_forward")
				layer_n = self.layers[i]
				mul_value = tf.matmul(prev_layer_activation, layer_n[0])
				add_value = tf.add(mul_value, layer_n[1])
				activation = tf.nn.relu(add_value)
				prev_layer_activation = activation

			#add dropout to the activation of the 2nd to last layer
			print("made it here!------------------")
			#prev_layer_activation = tf.nn.dropout(prev_layer_activation, self.dropout_prob)
			print(prev_layer_activation)
			print(self.layers[-1][0])
			op_mul = tf.matmul(prev_layer_activation, self.layers[-1][0])
			op_add = tf.add(op_mul, self.layers[-1][1])

		#change the last layer so that it doesn't do ReLU function
		with tf.name_scope("memrise"):
			memory = input_value
			self.memory_tensor = tf.concat([self.memory_tensor, memory], axis=0)
		return op_add
	def reminisce(self, memory_size):
		#sample generated images from memory to help prevent catastrophic forgetting
		max_val = tf.shape(self.memory_tensor)[0]

		indicies = tf.random_uniform([memory_size], minval=1, maxval=max_val, dtype=tf.int32)

		dreams = tf.gather(self.memory_tensor, indicies)

		dream_scores = self.forward_pass(dreams)

		return dream_scores


	def generator_loss(self, generated_examples):
		#toshiba
		print("--------generated_ex-----------")
		print(generated_examples)
		print("-------------------------------")
		#real_example_processing
		real_logits = self.forward_pass(self.real_examples)
		generator_logits = self.forward_pass(generated_examples)

		#remembering...
		mem_size = tf.shape(real_logits)[0]
		memory_logits = self.reminisce(mem_size)

		gen_length = tf.shape(generator_logits)[0]


		gen_labels = tf.tile(fake_one_hot, [gen_length])
		gen_labels = tf.reshape(gen_labels, [gen_length, 2])

		fake_labels = tf.tile(fake_one_hot, [mem_size])
		fake_labels = tf.reshape(fake_labels, [mem_size, 2])

		real_labels = tf.tile(real_one_hot, [mem_size])
		real_labels = tf.reshape(real_labels, [mem_size, 2])

		real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=real_labels)
		gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits, labels=gen_labels)
		mem_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=memory_logits, labels=fake_labels)

		total_loss = 0.5 * tf.add_n([real_loss, gen_loss, mem_loss])
		return tf.reduce_mean(total_loss)

	def network_variables(self):
		#it might not be the fastest way to go from rank 2 to 1 but it works
		op_var = []
		for chip in self.layers:
			op_var.extend(chip)
		return op_var

genos_layers = [[10, 500], [500, 500], [500, 784]]
genos = generator(genos_layers)


#making the discriminator and labels one-hot
#convention: [1, 0]=real; [0, 1]=fake
disky = discriminator([[784, 500], [500, 500], [500, 2]])

with tf.name_scope("disky_train"):
	forword = genos.forward_pass()
	print(forword)
	disc_loss = disky.generator_loss(forword)

	disky_optimizer = tf.train.AdamOptimizer().minimize(disc_loss, var_list=disky.network_variables())

	disky.memory_tensor = tf.concat([disky.memory_tensor, forword], 0)

with tf.name_scope("genos_train"):
	gen_pass = genos.forward_pass()
	disc_logits = disky.forward_pass(gen_pass)
	genos_loss = genos.discriminator_loss(disc_logits)

	print(genos.network_variables())
	genos_optimizer = tf.train.AdamOptimizer().minimize(genos_loss, var_list=genos.network_variables())

discriminator_special = tf.reduce_mean(tf.sigmoid(disc_logits))

def camera_ready(generated_img):
	generated_img = tf.nn.relu(generated_img)
	sig = tf.tanh(generated_img)
	scaled = tf.scalar_mul(255, sig)
	return tf.cast(scaled, tf.int32)

#Note: when training the discriminator use dropout but when training the generator with discriminator info disable dropout
#Note: remember to have the batch_size and minibatch_size values match up 

#note: there's a problem with running this on MNIST
#over time, the output of the generator becomes a black screen

#note: the discriminator isn't fooling the generator at all
with tf.Session() as sesh:
	init_op = tf.global_variables_initializer()
	sesh.run(init_op)
	train_writer = tf.summary.FileWriter('logs/adversarial_net_v3',
                                          sesh.graph)

	
	for i in range(100001):
		real_batch, _ = mnist.train.next_batch(minibatch_size)
		disc_opt, memo  = sesh.run([disky_optimizer, disky.memory_tensor], feed_dict={disky.dropout_prob: 1.0, disky.real_examples: real_batch})
		print(memo.shape)

		if i % k_disc_train_steps == 0:
			fishscale, _ = mnist.train.next_batch(minibatch_size)
			gen_opt, dsc, d_loss, g_loss, mem = sesh.run([genos_optimizer, discriminator_special, disc_loss, genos_loss, disky.memory_tensor], feed_dict={disky.dropout_prob: 1.0, disky.real_examples: fishscale})

			print("fool_level: " + str(d_loss))
			
			print("disc_loss: " + str(d_loss))
			print("genos_loss: " + str(g_loss))
			print()
		if i % 1000 == 0:
			group_size = 100
			last_pass = genos.forward_pass(batch_size=group_size)
			disc_curr_gen = disky.forward_pass(last_pass)

			cmra = camera_ready(last_pass)
			disc_curr_gen = tf.sigmoid(disc_curr_gen)

			output_set, disc_score = sesh.run([cmra, disc_curr_gen])
				
			top_index = np.argmax(disc_score)

			#disc_score = np.reshape(disc_score, (group_size))

			#top_array = np.argsort(disc_score, axis=0)

			#saver.save(sesh, file_path + "adversarial_model")

			print(disc_score)
			#print(output_set[top_index][np.argmax(output_set[top_index])])
			#print(top_array)
			for j in range(20):
				dir_name = "GAN_generated/adversarial_testing_" + str(i) + "/"

				if not os.path.exists(dir_name):
					os.mkdir(dir_name)

				output_img = output_set[j]

				print(cv2.imwrite(dir_name + "number_" + str(j) + ".png", np.reshape(output_img, (28, 28))))






