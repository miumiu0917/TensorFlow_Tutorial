import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main():
    # MNISTのダウンロード
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    # n * 784の可変2階テンソル
    x = tf.placeholder("float", [None, 784])
    
    # 重み行列とバイアスの宣言
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # ソフトマックス層の定義
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    
    # 正解用2階テンソルを用意
    y_ = tf.placeholder("float", [None, 10])
    
    # 誤差関数の交差エントロピー誤差関数を用意
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    
    # 学習方法を定義 0.01 学習率
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # 変数を初期化
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # 学習開始
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == "__main__":
    main()
