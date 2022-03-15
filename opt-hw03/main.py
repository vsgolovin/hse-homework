from methods import gradient_descent
import test_functions as tf


x = gradient_descent(tf.quadratic, tf.quadratic_derivatives, x0=(0, 0, 0, 0),
                     h_method=(0.1, 0.9))
print(x)


x = gradient_descent(tf.rosenbrock, tf.rosenbrock_derivatives, x0=(0, 0))
print(x)
