import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

def compute(x, y):
	c = tf.add(a, b)
	return c

with tf.Session() as session:
	v = session.run(compute(10, 20), feed_dict = {
		a : 10,
		b : 20
		})
	print(v)