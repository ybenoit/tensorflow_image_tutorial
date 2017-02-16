import tensorflow as tf

with tf.name_scope('counter'):
    # Counter Variable definition
    counter = tf.Variable(1, name="counter")
    tf.summary.scalar('counter', counter)

# Creation of a constant
two_op = tf.constant(2, name="const")

# Operations to perform in order to increment the variable value
new_value = tf.multiply(counter, two_op)
update = tf.assign(counter, new_value)

merged = tf.summary.merge_all()

# Initialize all variables
init_op = tf.global_variables_initializer()


with tf.Session() as sess:
    # Increment the value of the variable in a session
    sess.run(init_op)

    summary_writer = tf.summary.FileWriter("/tmp/nn_test", sess.graph)

    for i in range(5):
        summary, _ = sess.run([merged, update])
        summary_writer.add_summary(summary, i)
        print(sess.run(counter))

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# Multiplication operation
output = tf.multiply(input1, input2)

# Graph execution, we need to feed the placeholders
with tf.Session() as sess:
    result = sess.run(output, feed_dict={input1: [7.], input2: [2.]})
    print(result)
