# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
def cubic_equation():
    x = tf.constant([[100., 250., 854.]])
    y = tf.Variable(tf.zeros([1, 3]))
    a = tf.Variable(tf.zeros([1, 3]))
    b = tf.Variable(tf.zeros([1, 3]))
    c = tf.Variable(tf.zeros([1, 3]))
    d = tf.Variable(tf.zeros([1, 3]))
    # y = ax^3+bx^2+cx+d
    multiply = tf.multiply(x, y)
    multiply = tf.multiply(multiply, b)
    # multiply = tf.multiply(multiply, c)
    # multiply = tf.multiply(multiply, d)
    deviation = tf.square(y - multiply)

    train_step = tf.train.GradientDescentOptimizer(0.275).minimize(deviation)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(5000):
        sess.run(train_step)
    print(sess.run(x))
    print(sess.run(y))
    print(sess.run(b))
    print(sess.run(multiply))
# Here is the simple equation Y = X * Z
# while Z is a unknown value, X = (1, 2) and Y = (12, 4)
def simple_sum():
    x = tf.constant([[1., 2., 0.07]])
    y = tf.constant([[12., 8., 781]])
    Z = tf.Variable(tf.zeros([1, 3]))

    # yy = tf.add(x, Z)
    yy = tf.multiply(x, Z)
    deviation = tf.square(y - yy)

    trainer = tf.train.experimental
    train_step = tf.train.GradientDescentOptimizer(0.275).minimize(deviation)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(5000):
        sess.run(train_step)
    print(sess.run(Z))


if __name__ == "__main__":
    # simple_sum()
    cubic_equation()