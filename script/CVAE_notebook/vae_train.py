import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv3D, Conv3DTranspose, Flatten, Dense, Reshape, Lambda
)
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


# 1) The reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    batch   = K.shape(z_mean)[0]
    dim     = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# 2) Build encoder + decoder (no add_loss, no Functional‐API VAE)
def build_encoder_decoder(input_shape, latent_dim,
                          nlayers=2, base_filters=32,
                          kernel_size=3, intermediate_dim=128,
                          use_bias=True):
    # -- Encoder
    inputs = Input(shape=input_shape, name="encoder_input")
    x = inputs
    filters = base_filters
    for _ in range(nlayers):
        filters *= 2
        x = Conv3D(filters=filters,
                   kernel_size=kernel_size,
                   strides=2,
                   padding="same",
                   activation="relu",
                   use_bias=use_bias)(x)
    shape_info = K.int_shape(x)  # needed for decoder reshape

    x = Flatten()(x)
    x = Dense(intermediate_dim, activation="relu", use_bias=use_bias)(x)
    z_mean    = Dense(latent_dim, name="z_mean", use_bias=use_bias)(x)
    z_log_var = Dense(latent_dim, name="z_log_var", use_bias=use_bias)(x)
    z         = Lambda(sampling, name="z")([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

    # -- Decoder
    latent_inputs = Input(shape=(latent_dim,), name="z_sampling")
    x = Dense(intermediate_dim, activation="relu", use_bias=use_bias)(latent_inputs)
    total_dims = shape_info[1] * shape_info[2] * shape_info[3] * shape_info[4]
    x = Dense(total_dims, activation="relu", use_bias=use_bias)(x)
    x = Reshape((shape_info[1],
                 shape_info[2],
                 shape_info[3],
                 shape_info[4]))(x)

    filters = shape_info[4]
    for _ in range(nlayers):
        x = Conv3DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=2,
                            padding="same",
                            activation="relu",
                            use_bias=use_bias)(x)
        filters //= 2

    outputs = Conv3DTranspose(filters=input_shape[3],
                              kernel_size=kernel_size,
                              padding="same",
                              activation="sigmoid",
                              name="decoder_output",
                              use_bias=use_bias)(x)

    decoder = Model(latent_inputs, outputs, name="decoder")
    return encoder, decoder


# 3) Optional disentanglement helper
def compute_tc_and_disc(z, gamma):
    # batch‐split
    z_top, z_bottom = tf.split(z, 2, axis=0)
    # latent‐split
    z1_a, z1_b = tf.split(z_top,    2, axis=1)
    z2_a, z2_b = tf.split(z_bottom, 2, axis=1)

    # build positive / negative pairs
    q     = tf.concat([tf.concat([z1_b, z1_b], axis=1),
                       tf.concat([z2_b, z2_b], axis=1)],
                      axis=0)
    q_bar = tf.concat([tf.concat([z1_b, z2_b], axis=1),
                       tf.concat([z2_b, z1_b], axis=1)],
                      axis=0)

    discriminator = tf.keras.Sequential([
        Dense(1, activation="sigmoid")
    ], name="discriminator")

    eps = 1e-7
    q_score_raw     = discriminator(q)
    q_bar_score_raw = discriminator(q_bar)

    q_score     = tf.clip_by_value((q_score_raw     + 0.1) * 0.85, eps, 1-eps)
    q_bar_score = tf.clip_by_value((q_bar_score_raw + 0.1) * 0.85, eps, 1-eps)

    # total-correlation term
    tc_loss = tf.reduce_mean(tf.math.log(q_score / (1 - q_score)))
    # discriminator cross‐entropy term
    disc_loss = tf.reduce_mean(-tf.math.log(q_score)
                               - tf.math.log(1 - q_bar_score))

    return gamma * tc_loss, disc_loss


# 4) The subclassed VAE
class VAE(tf.keras.Model):
    def __init__(self,
                 encoder,
                 decoder,
                 image_size,
                 channels,
                 gamma=1.0,
                 disentangle=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.image_size = image_size
        self.channels   = channels
        self.gamma      = gamma
        self.disentangle = disentangle

        # Trackers
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker    = tf.keras.metrics.Mean(name="kl_loss")
        if disentangle:
            self.tc_loss_tracker   = tf.keras.metrics.Mean(name="tc_loss")
            self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        m = [self.total_loss_tracker,
             self.recon_loss_tracker,
             self.kl_loss_tracker]
        if self.disentangle:
            m += [self.tc_loss_tracker, self.disc_loss_tracker]
        return m

    def train_step(self, data):
        x = data  # expecting shape (batch, X,Y,Z,channels)
        with tf.GradientTape() as tape:
            # 1) encode
            z_mean, z_log_var, z = self.encoder(x, training=True)
            # 2) decode
            x_recon = self.decoder(z, training=True)

            # 3) reconstruction loss (MSE × volume)
            vol_size = (self.image_size ** 3) * self.channels
            recon_loss = tf.reduce_mean(tf.square(x - x_recon)) * vol_size

            # 4) KL divergence
            kl = -0.5 * (1 + z_log_var
                         - tf.square(z_mean)
                         - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl, axis=1))

            total_loss = recon_loss + kl_loss

            # 5) (optional) disentanglement
            if self.disentangle:
                tc, disc = compute_tc_and_disc(z, gamma=self.gamma)
                total_loss += tc + disc

        # 6) backprop
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # 7) update trackers
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        if self.disentangle:
            self.tc_loss_tracker.update_state(tc)
            self.disc_loss_tracker.update_state(disc)

        return {m.name: m.result() for m in self.metrics}


# 5) Putting it all together
if __name__ == "__main__":
    # a) build encoder+decoder
    input_shape = (64, 64, 64, 1)
    latent_dim  = 32

    encoder, decoder = build_encoder_decoder(
        input_shape=input_shape,
        latent_dim=latent_dim,
        nlayers=2,
        base_filters=32,
        intermediate_dim=128
    )

    # b) instantiate the subclassed VAE
    vae = VAE(
        encoder=encoder,
        decoder=decoder,
        image_size=input_shape[0],
        channels=input_shape[3],
        gamma=100.0,
        disentangle=True,
        name="mri_vae_3d"
    )

    # c) compile & train
    vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    # assume `train_dataset` yields batches of shape (batch,64,64,64,1)
    # vae.fit(train_dataset, epochs=50, batch_size=32)