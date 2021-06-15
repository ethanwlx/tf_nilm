import tensorflow as tf


class ScaledDotProductAttention(tf.keras.layers.Layer):

    def __init__(self, attention_dropout_rate=0.0, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(attention_dropout_rate, name='dropout')

    def call(self, query, key, value, training, mask=None):
        query = tf.cast(query, dtype=self.dtype)
        key = tf.cast(key, dtype=self.dtype)
        value = tf.cast(value, dtype=self.dtype)

        score = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(query)[-1], tf.float32)
        score = score / tf.math.sqrt(dk)
        if mask is not None:
            attention = tf.cast(attention, dtype=self.dtype)
            score += (1.0 - attention) * -10000.0
        attn_weights = tf.nn.relu(score)
        attn_weights = self.dropout(attn_weights, training=training)
        context = tf.matmul(attn_weights, value)
        return context, attn_weights


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model=1024, num_heads=8, attention_dropout_rate=0.1, epsilon=1e-8, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        assert self.d_model % self.num_heads == 0

        self.query_weight = tf.keras.layers.Dense(self.d_model, name='query')
        self.key_weight = tf.keras.layers.Dense(self.d_model, name='key')
        self.value_weight = tf.keras.layers.Dense(self.d_model, name='value')

        self.attention = ScaledDotProductAttention(attention_dropout_rate=attention_dropout_rate, name='selfattention')

        # # output block
        self.dense = tf.keras.layers.Dense(self.d_model, name='dense')

    def _split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_model // self.num_heads))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, mask=None, training=None):
        origin_input = query  # query == key == value

        batch_size = tf.shape(query)[0]
        query = self._split_heads(self.query_weight(query), batch_size)
        key = self._split_heads(self.key_weight(key), batch_size)
        value = self._split_heads(self.value_weight(value), batch_size)

        context, attn_weights = self.attention(query, key, value, mask=None)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, [batch_size, -1, self.d_model])
        output = self.dense(context)
        return output, attn_weights


class ConvLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(ConvLayer, self).__init__()
        self.Conv = tf.keras.layers.Conv1D(filters=1, kernel_size=3, padding='causal')
        self.norm = tf.keras.layers.BatchNormalization(axis=1)
        self.activation = tf.nn.elu
        self.maxPool = tf.keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')

    def call(self, x):
        x = self.Conv(tf.transpose(x, perm[0, 2, 1]))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = tf.transpose(x, perm[1, 2])
        return x


class PointWiseFeedForwardNetwork(tf.keras.layers.Layer):

    def __init__(self, d_model=512, **kwargs):
        super(PointWiseFeedForwardNetwork, self).__init__(**kwargs)
        self.d_model = d_model
        self.ffn_size = d_model * 4
        self.dense1 = tf.keras.layers.Dense(self.ffn_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.d_model)

    def call(self, inputs, training=None):
        outputs = self.dense2(self.dense1(inputs))
        return outputs

    def get_config(self):
        config = {
            'ffn_size': self.ffn_size,
            'd_model': self.d_model,
        }
        p = super(PointWiseFeedForwardNetwork, self).get_config()
        return dict(list(p.items()) + list(config.items()))


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, rate=0.1, epsilon=1e-8, **kwargs):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_size = d_model * 4
        self.dropout_rate = rate
        self.epsilon = epsilon

        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
        self.mha_dropout = tf.keras.layers.Dropout(rate)
        self.mha_layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)

        self.ffn = PointWiseFeedForwardNetwork(self.d_model)
        self.ffn_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.ffn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)

    def call(self, x, training, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.mha_dropout(attn_output, training=training)
        attn_output = self.mha_layer_norm(x + attn_output)

        ffn_output = self.ffn(attn_output)
        ffn_output = self.ffn_dropout(ffn_output, training=training)
        ffn_output = self.ffn_layer_norm(ffn_output + attn_output)

        return ffn_output

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ffn_size': self.ffn_size,
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon
        }
        base = super().get_config()
        return dict(list(base.items()) + list(config.items()))


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, window_size, rate=0.1, epsilon=1e-8, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        self.enc_layers = [
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                rate=rate,
                epsilon=epsilon,
                name='layer_{}'.format(i)
            ) for i in range(num_layers)
        ]

        # self.ecov_layers = [ConvLayer() for i in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        batch_size_ = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size_, -1, 1])
        # adding embedding and position encoding.

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
            # x = self.ecov_layers[i](x)
        # print("output shape is {}".format(x.shape))

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, d_model, window_size, rate=0.0):
        super(Decoder, self).__init__()

        self.window_size = window_size
        self.conv1 = tf.keras.layers.Conv1D(20, 8, activation='relu', input_shape=(window_size, d_model),
                                            padding="same", strides=1)
        self.conv2 = tf.keras.layers.Conv1D(20, 6, activation='relu', padding="same", strides=1)
        self.conv3 = tf.keras.layers.Conv1D(30, 5, activation='relu', padding="same", strides=1)
        self.conv4 = tf.keras.layers.Conv1D(40, 4, activation='relu', padding="same", strides=1)
        self.conv5 = tf.keras.layers.Conv1D(40, 4, activation='relu', padding="same", strides=1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='linear')
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        # x = tf.reshape(x,[batch_size,-1])
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    # def get_config(self):
    #     config = {
    #         'd_model': self.d_model,

    #     }
    #     base = super(DecoderLayer, self).get_config()
    #     return dict(list(base.items()) + list(config.items()))


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, window_size, rate=0.1):
        super(Transformer, self).__init__()
        self.window_size = window_size
        self.encoder = Encoder(num_layers, d_model, num_heads, window_size, rate)
        self.decoder = Decoder(d_model, window_size)

    def call(self, inp, training, mask=None):
        # inp = tf.reshape(inp,[-1,self.window_size,1])


        enc_output = self.encoder(inp, training, mask)  # (batch_size, inp_seq_len, d_model)
        # print(enc_output.shape)
        dec_output = self.decoder(enc_output, training)
        # dec_output = self.decoder(inp, training)
        # print(dec_output.shape)
        return dec_output

class PatchesEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchesEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        patch_size = seq_length//self.num_patches

        
        x = tf.reshape(x, [batch_size, self.num_patches, patch_size])

        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(x) + self.position_embedding(positions)
              
        return encoded

class VitEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, num_patches, rate=0.1, epsilon=1e-8, **kwargs):
        super(VitEncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_patches = num_patches
        self.ffn_size = d_model*4
        self.dropout_rate = rate
        self.epsilon = epsilon
        self.patchesencoder = PatchesEncoder(self.num_patches, self.d_model)
        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
        self.mha_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.mha_layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)        
        
        self.ffn = PointWiseFeedForwardNetwork(self.d_model)
        self.ffn_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.ffn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.epsilon)        



    def call(self, x, training, mask=None):
        x = self.patchesencoder(x)
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.mha_dropout(attn_output, training=training)
        attn_output = self.mha_layer_norm(x+ attn_output)
        
        ffn_output = self.ffn(attn_output)
        ffn_output = self.ffn_dropout(ffn_output, training=training)
        ffn_output = self.ffn_layer_norm(ffn_output + attn_output)

        return ffn_output
  
class VitEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, window_size, rate=0.1, epsilon=1e-8, **kwargs):
        super(VitEncoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.enc_layers = [
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                num_patches=window_size,
                rate=rate,
                epsilon=epsilon,
                name='layer_{}'.format(i)
            ) for i in range(num_layers)
        ]

        #self.ecov_layers = [ConvLayer() for i in range(num_layers)]   

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):

        batch_size_ = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size_, -1 , 1])
        # adding embedding and position encoding.
        
        for i in range(self.num_layers):
          x = self.enc_layers[i](x, training, mask)
          #x = self.ecov_layers[i](x)
        #print("output shape is {}".format(x.shape))

        return x  # (batch_size, input_seq_len, d_model)

class VitTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, window_size, rate=0.1):
        super(VitTransformer, self).__init__()
        self.window_size = window_size
        self.encoder = VitEncoder(num_layers, d_model, num_heads, window_size, rate)
        self.decoder = Decoder(d_model, window_size)
    def call(self, inp, training, mask=None):
        # inp = tf.reshape(inp,[-1,self.window_size,1])

        enc_output = self.encoder(inp, training, mask)  # (batch_size, inp_seq_len, d_model)
        #print(enc_output.shape)
        dec_output = self.decoder(enc_output, training)
        # dec_output = self.decoder(inp, training)
        #print(dec_output.shape)
        return dec_output

class s2p(tf.keras.Model):
    def __init__(self, window_size, d_model=1, rate=0.0):
        super(s2p, self).__init__()

        self.window_size = window_size
        self.conv1 = tf.keras.layers.Conv1D(20, 8, activation='relu', input_shape=(window_size, d_model),
                                            padding="same", strides=1)
        self.conv2 = tf.keras.layers.Conv1D(20, 6, activation='relu', padding="same", strides=1)
        self.conv3 = tf.keras.layers.Conv1D(30, 5, activation='relu', padding="same", strides=1)
        self.conv4 = tf.keras.layers.Conv1D(40, 4, activation='relu', padding="same", strides=1)
        self.conv5 = tf.keras.layers.Conv1D(40, 4, activation='relu', padding="same", strides=1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='linear')
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x,[batch_size,-1,1])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x