import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, (3, 3), padding= "same", strides=1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.maxpool1 = tf.keras.layers.MaxPool2D((2, 2))

    @tf.function()
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x, training=training)
        x = self.maxpool1(x)
        return x

class CharacterRecognition(tf.keras.Model):
    def __init__(self):
        super(CharacterRecognition, self).__init__()
        self.block1 = ConvBlock(32)
        self.block2 = ConvBlock(64)

        self.dense0 = tf.keras.layers.Dense(512, activation= "relu")

        self.dense1 = tf.keras.layers.Dense(256, activation= "relu")

        #self.dropout = tf.keras.layers.Dropout(0.5)

        self.dense2 = tf.keras.layers.Dense(31)
        self.flat = tf.keras.layers.Flatten()

    @tf.function()
    def call(self, x, training=False):
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.flat(x)
        x = self.dense0(x)
        #x = self.dropout(x)
        x = self.dense1(x)
        #x = self.dropout(x)
        x = self.dense2(x)
        return x
