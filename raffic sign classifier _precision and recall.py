import pickle
import numpy as np

# Loading pickled data
training_file= 'D://PyCharm_Python//Summer of science//kaggle//train.p'
validation_file= 'D://PyCharm_Python//Summer of science//kaggle//valid.p'
testing_file= 'D://PyCharm_Python//Summer of science//kaggle//test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train_pure, y_train_pure = train['features'], train['labels']
X_valid_pure, y_valid = valid['features'], valid['labels']
X_test_pure, y_test = test['features'], test['labels']


# Summarizing the data

n_train = len(X_train_pure)
n_valid=len(X_valid_pure)
n_test=len(X_test_pure)

# Traffic labels present
n_classes = len(np.unique(y_train_pure))
# Each trainng image is of size:
image_shape=X_train_pure[0].shape

import matplotlib.pyplot as plt
import random

# plot class distribution
num_of_samples = []
for i in range(n_classes):
    num_samples_class =(y_train_pure == i).sum()
    num_of_samples.append(num_samples_class)

plt.figure(figsize=(12,4))
plt.bar(range(n_classes), num_of_samples)
plt.title('Distribution of training data')
plt.xlabel('Class number')
plt.ylabel('Number of images')
plt.show()

### Data preprocess ###
import cv2

def equal_Hist(img):
    # Equalization Histogram
    img_temp = img.copy()
    img_temp[:,:,0] = cv2.equalizeHist(img[:, :, 0])
    img_temp[:,:,1] = cv2.equalizeHist(img[:, :, 1])
    img_temp[:,:,2] = cv2.equalizeHist(img[:, :, 2])
    return img_temp

def crop(img, margin=0):
    return img[margin:(img.shape[0]-margin), margin:(img.shape[1]-margin)]

def blur(img):
    gb = cv2.GaussianBlur(img, (5,5), 20.0)
    return cv2.addWeighted(img, 2, gb, -1, 0)

def process(img):
    result = crop(img, margin=2)
    result = blur(result)
    result = equal_Hist(result)
    return result


def rotate_img(img):
    mid_x, mid_y = int(img.shape[0]/2), int(img.shape[1]/2)
    ang = 30.0*np.random.rand()-15
    Mat = cv2.getRotationMatrix2D((mid_x, mid_y), ang, 1.0)
    return cv2.warpAffine(img, Mat, img.shape[:2])

# Processing and plotting training data

def plotimage(img):
    img1 = crop(img, margin=2)
    img2 = blur(img1)
    img3 = equal_Hist(img2)

    plt.figure(figsize=(8, 2))
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title("Raw image({}x{})".format(img.shape[0], img.shape[1]))
    plt.subplot(1, 4, 2)
    plt.imshow(img1)
    plt.title("crop image({}x{})".format(img1.shape[0], img1.shape[1]))
    plt.subplot(1, 4, 3)
    plt.imshow(img2)
    plt.title("Sharpen")
    plt.subplot(1, 4, 4)
    plt.imshow(img3)
    plt.title("eqHist")

#plotting 4 random images
for num in range(4):
    index = random.randint(0, len(X_train_pure))
    plotimage(X_train_pure[index,:,:,:])
    plt.show()

#### data augmentation by rotating ####
ylabel_small = []
for i in range(43):
    if num_of_samples[i] <= 400:
        ylabel_small.append(i)
print(len(ylabel_small),' Labels whose quantity is less than 400 are',ylabel_small)


X_train_new, X_train = [], []
y_train_new, y_train = [], []
for idx in range(len(X_train_pure)):
    img = X_train_pure[idx]
    X_train.append(process(img))
    if y_train_pure[idx] in ylabel_small:
        X_train_new.append(rotate_img(process(img)))
        y_train_new.append(y_train_pure[idx])

print('Labels whose quantity less than 400 =', ylabel_small)
print('new data number =', len(X_train_new))
print('origin train data number = ', len(X_train_pure))

X_train = np.append(X_train, X_train_new, axis=0)
y_train = np.hstack((y_train_pure, y_train_new))

print('Updated train data number =', len(X_train))
print('Updated train label number =', len(y_train))

# Processing validation data
X_valid = []
for idx in range(len(X_valid_pure)):
    img = X_valid_pure[idx]
    X_valid.append(process(img))
print('Updated Valid Image shape: {}'.format(X_valid[0].shape))

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)
print("Updated Train Image Shape: {}".format(X_train[0].shape))


### Model Architcture ###

import tensorflow as tf
from tensorflow.contrib.layers import flatten

EPOCHS = 30
BATCH_SIZE = 128
def Sign(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    ConvStrides = [1, 1]
    PoolStrides = [2, 2]
    L1Filter = [5, 5, 3] # [Filter height, Filter width, Input depth]
    L1Output = [(28-L1Filter[0]+1) / ConvStrides[0], (28-L1Filter[1]+1) / ConvStrides[1], 16]  # VALID Padding output
    # computation formula
    L2Filter = [3, 3, L1Output[2]]
    L2Output = [(L1Output[0]/PoolStrides[0] - L2Filter[0] + 1)/ConvStrides[0],
                (L1Output[1]/PoolStrides[1] - L2Filter[1] + 1)/ConvStrides[1], 30]
    L3Filter = [2, 2, L2Output[2]]
    L3Output = [(L2Output[0]/PoolStrides[0] - L3Filter[0] + 1)/ConvStrides[0],
                (L2Output[1]/PoolStrides[1] - L3Filter[1] + 1)/ConvStrides[1], 60]
    L4Input = int((L3Output[0]/PoolStrides[0]) * (L3Output[1]/PoolStrides[1]) * L3Output[2])
    L4Output = 160
    L5Output = 80


    # Layer 1: Convolutional.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(L1Filter[0], L1Filter[1], L1Filter[2], L1Output[2]), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(L1Output[2]))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, ConvStrides[0], ConvStrides[1], 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, PoolStrides[0], PoolStrides[1], 1], padding='VALID')

    # Layer 2: Convolutional.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(L2Filter[0], L2Filter[1], L2Filter[2], L2Output[2]), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(L2Output[2]))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, ConvStrides[0], ConvStrides[1], 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, PoolStrides[0], PoolStrides[1], 1], padding='VALID')

    # Layer 3: Convolutional.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(L3Filter[0], L3Filter[1], L3Filter[2], L3Output[2]), mean=mu, stddev=sigma))
    conv3_b = tf.Variable(tf.zeros(L3Output[2]))
    conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, ConvStrides[0], ConvStrides[1], 1], padding='VALID') + conv3_b

    # Activation.
    conv3 = tf.nn.relu(conv3)

    # Pooling.
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, PoolStrides[0], PoolStrides[1], 1], padding='VALID')

    # Flatten. Input = 2x2x60. Output = 240.
    fc0 = flatten(conv3)

    # Layer 4: Fully Connected. Input = 240. Output = 160.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(L4Input, L4Output), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(L4Output))
    fc0 = tf.nn.dropout(fc0, keep_prob)
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 5: Fully Connected. Input = 160. Output = 80.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(L4Output, L5Output), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(L5Output))
    # fc1 = tf.nn.dropout(fc1, keep_prob)
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 6: Fully Connected. Input = 80. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(L5Output, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    # fc2 = tf.nn.dropout(fc2, keep_prob)
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

##training and performance evaluation on test sets

x = tf.placeholder(tf.float32, (None, 28, 28, 3))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)

rate = 0.001

logits = Sign(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples




with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    AccuracySet = []
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.65})

        training_accuracy = evaluate(X_train, y_train)
        print("EPOCH {} ...".format(i + 1))
        print("Training Accuracy = {:.2f}%".format(training_accuracy * 100))

        validation_accuracy = evaluate(X_valid, y_valid)
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        AccuracySet.append(validation_accuracy * 100)
    saver.save(sess, './Traffic-Sign-CNNmodel')
    print("Model saved")

# Pre-process the test images
test_imgs = []
for idx in range(len(X_test_pure)):
    img = X_test_pure[idx]
    test_imgs.append(process(img))

print("Test image shape = {}".format(test_imgs[0].shape))

def eval_performance(X_data, y_data, BATCH_SIZE=1):
    num_examples = len(X_data)
    tp = np.zeros(n_classes)
    fp = np.zeros(n_classes)
    fn = np.zeros(n_classes)

    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        pred = sess.run(tf.argmax(logits, 1), feed_dict={x: batch_x, keep_prob: 1})
        real = batch_y
        out = sess.run(tf.equal(pred, real))
        for i in range(len(out)):
            if out[i]:
                tp[pred[i]] += 1
            else:
                fp[pred[i]] += 1
                fn[real[i]] += 1

    precision = [tp[i] / (tp[i] + fp[i]) if tp[i] != 0 else 0 for i in range(len(tp))]
    recall = [tp[i] / (tp[i] + fn[i]) if tp[i] != 0 else 0 for i in range(len(tp))]
    return precision, recall
########################

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(test_imgs, y_test)
    test_precision, test_recall = eval_performance(test_imgs, y_test, BATCH_SIZE)

    print("Test Accuracy = {:.3f}".format(test_accuracy))

labels = np.asarray([1.*i for i in range(n_classes)])
########################
fig = plt.figure(figsize=(15, 5))
ax = plt.subplot(1,1,1)
l1=ax.bar(labels-0.15, test_precision, width=0.2, facecolor='g', alpha=0.65)
l2=ax.bar(labels+0.15, test_recall, width=0.2, facecolor='r', alpha=0.65)
ax.set_title('Precision and Recall on Test set')
ax.set_xlabel('Labels')
ax.set_xticks(range(n_classes))
plt.legend([l1, l2],["Precision", "Recall"]);
