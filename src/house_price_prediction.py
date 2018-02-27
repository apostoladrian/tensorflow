import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# generate house sizes between 1000 and 35000
number_of_houses = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=number_of_houses)

# generate house prices
np.random.seed(54)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000,size=number_of_houses)

# plot the generated house size and price
# plt.plot(house_size, house_price, "bx")
# plt.ylabel("Price")
# plt.xlabel("Size")
# plt.show()

# print(house_size)
# print(house_price)

# we need to normalize the values to prevent under/overflow
def normalize(array):
    return (array - array.mean()) / array.std()

# define number of training samples - 0.7
num_train_samples = int(math.floor(number_of_houses * 0.7))
print(num_train_samples)

# define the training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asarray(house_price[:num_train_samples])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# print(train_house_size_norm)
# print(train_house_size.mean())
# print(train_house_size.std())

# define test data
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])

# setup the tensorflow placeholders
tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="price")

# define the variables holding the size_factor and price_offset
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

# define the operations for predicting values
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

# define the loss function - mean squared error
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_price, 2))/(2 * num_train_samples)

# define the learning rate
learning_rate = 0.1

# define a gradient descent optimizer that will minimize the loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# initialize the variables for tf
init = tf.global_variables_initializer()

# launch the graphin the session
with tf.Session() as sess:
    sess.run(init)

    # set how often we will display training p[rogress
    display_every = 2
    num_training_iter = 50

    # calculate the number of lines to animation
    fit_num_plots = int(math.floor(num_training_iter/display_every))
    # add storage of factor and offset values from each epoch
    fit_size_factor = np.zeros(fit_num_plots)
    fit_price_offsets = np.zeros(fit_num_plots)
    fit_plot_idx = 0    

    #keep iterating the training data
    for iteration in range(num_training_iter):
        # fit the traning data
        for (x, y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

        #display current training status
        if (iteration + 1) % display_every == 0:
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
            print(" iteration #: ", "%04d" % (iteration + 1), "cost=", "{:.9f}".format(c), "size_factor=", \
                sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))

            # Save the fit size_factor and price_offset to allow animation of learning process
            fit_size_factor[fit_plot_idx] = sess.run(tf_size_factor)
            fit_price_offsets[fit_plot_idx] = sess.run(tf_price_offset)
            fit_plot_idx = fit_plot_idx + 1

    print("Optimization Finished!")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
    print("Trainer cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset), "\n")

    # get values used to normalized data so we can denormalize data back to its original scale
    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_price.mean()
    train_price_std = train_price.std()

    # Plot the graph
    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    # training data
    plt.plot(train_house_size, train_price, 'go', label='Training data')
    # testing data
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')

    # learned regression
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
             (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean,
             'k-',
             label='Learned Regression')
 
    plt.legend(loc='upper left')
    plt.show()


    # 
    # Plot another graph that animation of how Gradient Descent sequentually adjusted size_factor and price_offset to 
    # find the values that returned the "best" fit line.
    fig, ax = plt.subplots()
    line, = ax.plot(house_size, house_price)

    plt.rcParams["figure.figsize"] = (10,8)
    plt.title("Gradient Descent Fitting Regression Line")
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size, train_price, 'go', label='Training data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')

    def animate(i):
        line.set_xdata(train_house_size_norm * train_house_size_std + train_house_size_mean)  # update the data
        line.set_ydata((fit_size_factor[i] * train_house_size_norm + fit_price_offsets[i]) * train_price_std + train_price_mean)  # update the data
        return line,
 
     # Init only required for blitting to give a clean slate.
    def initAnim():
        line.set_ydata(np.zeros(shape=house_price.shape[0])) # set y's to 0
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, fit_plot_idx), init_func=initAnim,
                                 interval=1000, blit=True)

    plt.show()  