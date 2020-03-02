from __future__ import absolute_import, division, print_function, unicode_literals

import time

import tensorflow as tf

from Codigo.transformer.transformer_aux import create_masks, loss_function
from layers.decoder import Decoder
from layers.encoder import Encoder


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,  # input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.optimizer_custom = None
        self.loss_custom = None
        self.metrics_custom = None
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    def compile(self,
                optimizer,
                loss,
                metrics
                ):
        self.optimizer_custom = optimizer
        self.loss_custom = loss
        self.metrics_custom = metrics

    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.float64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    # @tf.function(input_signature=train_step_signature)
    @tf.function()
    def train_step(self, inp, tar):
        # Comentado porque esto es para que el decoder no vea la salida, pero como nuestra
        # salida es solamente una unidad no lo voy a usar de momento.

        # tar_inp = tar[:, :-1]
        # tar_real = tar[:, 1:]
        tar_inp = tar
        tar_real = tar

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self.call(inp, tar_inp,
                                       True,
                                       enc_padding_mask,
                                       combined_mask,
                                       dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer_custom.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_custom(loss)
        self.metrics_custom(tar_real, predictions)

    def fit(self, train_dataset, epochs, checkpoint_path="./checkpoints/train"):

        ckpt = tf.train.Checkpoint(transformer=self,
                                   optimizer=self.optimizer_custom)

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        for epoch in range(epochs):
            start = time.time()

            self.loss_custom.reset_states()
            self.metrics_custom.reset_states()

            for (batch, (inp, tar)) in enumerate(train_dataset):
                self.train_step(inp, tar)

                if batch % 50 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, self.loss_custom.result(), self.metrics_custom.result()))

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                self.loss_custom.result(),
                                                                self.metrics_custom.result()))

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    def evaluate(self, inp_sentence, max_length=1):

        encoder_input = tf.expand_dims(inp_sentence, 0)

        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [0]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.call(encoder_input,
                                                       output,
                                                       False,
                                                       enc_padding_mask,
                                                       combined_mask,
                                                       dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == 3:
                return tf.squeeze(output, axis=0), attention_weights

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights


class Transformers_encoding(Transformer):
    def fit(self, train_dataset, epochs, checkpoint_path="./checkpoints/train"):
        ckpt = tf.train.Checkpoint(transformer=self,
                                   optimizer=self.optimizer_custom)

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        for epoch in range(epochs):
            start = time.time()

            self.loss_custom.reset_states()
            self.metrics_custom.reset_states()

            for (batch, (inp, tar)) in enumerate(train_dataset):
                self.train_step(inp, tar)

                if batch % 50 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, self.loss_custom.result(), self.metrics_custom.result()))

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                self.loss_custom.result(),
                                                                self.metrics_custom.result()))

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
