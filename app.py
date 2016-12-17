from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow;

pixel_count = 28 * 28
number_count = 10

# TF placeholders
x = tensorflow.placeholder(tensorflow.float32, shape=[None, pixel_count])
y_ = tensorflow.placeholder(tensorflow.float32, shape=[None, number_count])

# TF variables
Weights = tensorflow.Variable(tensorflow.zeros([pixel_count, number_count]), name='Weights')
bias = tensorflow.Variable(tensorflow.zeros([number_count]), name='bias')
training_step = tensorflow.Variable(0, name='training_step')

# classifier function
y = tensorflow.nn.softmax(tensorflow.matmul(x, Weights) + bias)
# loss function
cross_entropy = tensorflow.reduce_mean(-tensorflow.reduce_sum(y_ * tensorflow.log(y), reduction_indices=[1]))

# training Dataset
x_train = mnist.train.images # images
y_train = mnist.train.labels # values on images
# testing Dataset
x_test = mnist.test.images # images
y_test = mnist.test.labels # values on images

# training cycle properties
LEARNING_RATE = 0.1
TRAIN_STEPS = 100

# training method for classifier
training = tensorflow.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

correct_prediction = tensorflow.equal(tensorflow.argmax(y, 1), tensorflow.argmax(y_, 1))
accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))

# create session and RUN!!
init = tensorflow.global_variables_initializer();
with tensorflow.Session() as session:
    session.run(init);

    # restore variables
    saver = tensorflow.train.Saver()
    try:
        saver.restore(session, 'model/mnist-model')
        print('Model restored')
    except Exception as error:
        print('No model available')
        pass

    print('Start training step: ' + str(session.run(training_step)))

    for i in range(session.run(training_step), session.run(training_step) + TRAIN_STEPS):
        session.run(training, feed_dict={ x: x_train, y_: y_train })
        if i > 0 and i % 10 == 0:
            loss_ = session.run(cross_entropy, { x: x_train, y_: y_train })
            accuracy_ = session.run(accuracy, { x: x_test, y_: y_test })
            print('Training step: ' + str(i) + ', Accuracy = ' + str(accuracy_) + ', Loss = ' + str(loss_))

    session.run(training_step.assign(training_step + TRAIN_STEPS))
    print('End training step: ' + str(session.run(training_step)))
    saver.save(session, 'model/mnist-model')
