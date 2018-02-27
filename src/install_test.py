#import tensorflow
import tensorflow as tf

#start a session
sess = tf.Session()

# verify we cna print a string
hello = tf.constant("Hellow world from Tensorflow")
print(sess.run(hello))

#perform some simple math
a = tf.constant(20)
b = tf.constant(15)

print("a + b = {0}".format(sess.run(a + b)))