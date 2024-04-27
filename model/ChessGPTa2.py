import keras
import tensorflow as tf
import config
import keras_nlp
import math
from keras_nlp.layers import TransformerDecoder
from keras_nlp.layers import TokenAndPositionEmbedding
from keras_nlp.layers import SinePositionEncoding
from preprocess.strategies import color_masking
from keras_nlp.layers import RotaryEmbedding


# Small Settings
# dense_dim = config.dense_dim_small
# heads = config.heads_small
# embed_dim = config.embed_dim_small


# Regular Settings
dense_dim = config.dense_dim
heads = config.heads
embed_dim = config.embed_dim




@keras.saving.register_keras_serializable(package="ChessGPTa2", name="ChessGPTa2")
class ChessGPTa2(tf.keras.Model):
    def __init__(self):
        super().__init__(name='ChessGPTa2')
        self.m_type = 'a2'
        self.supports_masking = True
        self.positional = True
        self.dense_dim = dense_dim
        self.num_heads = heads
        self.embed_dim = embed_dim

        # Move Embeddings
        self.embedding_layer = keras.layers.Embedding(
            config.vocab_size,
            self.embed_dim,
            mask_zero=True
        )
        self.positional_embedding = RotaryEmbedding()

        # Decoder Stack
        self.norm_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)
        self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)
        self.decoder_3 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)
        self.decoder_4 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)


        # Move Prediction Head
        self.move_prediction_head = keras.layers.Dense(
            config.vocab_size,
            name="move_prediction_head",
            activation="linear",
        )

        # Value Prediction Head
        self.value_prediction_head = keras.layers.Dense(
            1,
            name="value_prediction_head",
            activation="linear",
        )



    def call(self, inputs, training=False):

        # Inputs
        model_moves, opp_moves, is_white = inputs

        # Embed model moves
        model_move_embeddings = self.embedding_layer(model_moves)
        model_move_embeddings = self.positional_embedding(model_move_embeddings)

        # Embed opponent moves
        opp_move_embeddings = self.embedding_layer(opp_moves)
        opp_move_embeddings = self.positional_embedding(opp_move_embeddings)

        # Get causal mask for cross attention
        causal_cross_mask = color_masking.generate_batch_of_causal_matrices(is_white)

        # Decoder Stack
        decoded_move = model_move_embeddings
        decoded_move = self.decoder_1(
            decoded_move,
            encoder_sequence=opp_move_embeddings,
            encoder_attention_mask=causal_cross_mask,
            use_causal_mask=True,
            training=training
        )
        decoded_move = self.decoder_2(
            decoded_move,
            encoder_sequence=opp_move_embeddings,
            encoder_attention_mask=causal_cross_mask,
            use_causal_mask=True,
            training=training
        )
        decoded_move = self.decoder_3(
            decoded_move,
            encoder_sequence=opp_move_embeddings,
            encoder_attention_mask=causal_cross_mask,
            use_causal_mask=True,
            training=training
        )
        decoded_move = self.decoder_4(
            decoded_move,
            encoder_sequence=opp_move_embeddings,
            encoder_attention_mask=causal_cross_mask,
            use_causal_mask=True,
            training=training
        )

        # Move Prediction Head
        move_predictions = self.move_prediction_head(decoded_move)

        # Value Prediction Head
        value_predictions = self.value_prediction_head(decoded_move)


        return move_predictions, value_predictions


    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    pt_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True, ignore_class=0)
    pt_loss_tracker = tf.keras.metrics.Mean(name="loss")
    pt_perplexity_tracker = keras_nlp.metrics.Perplexity(name="perplexity", from_logits=True, mask_token_id=0)


    def train_step(self, inputs):
        model_inputs, model_labels, cross_inputs, is_white = inputs
        m_inputs = [model_inputs, cross_inputs, is_white]
        with tf.GradientTape() as tape:
            # Forward Pass
            predictions, val_predcitions = self(m_inputs, training=True)
            buloss = self.pt_loss_fn(model_labels, predictions)

            # DISTRIBUTED TRAINING
            if config.distributed is True:
                uloss = tf.nn.compute_average_loss(buloss, global_batch_size=config.global_batch_size)
            else:
                uloss = buloss

            # Mixed Precision
            if config.mixed_precision is True:
                loss = self.optimizer.get_scaled_loss(uloss)
            else:
                loss = uloss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        if config.mixed_precision is True:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.pt_loss_tracker.update_state(buloss)
        self.pt_perplexity_tracker.update_state(model_labels, predictions)

        return {"loss": self.pt_loss_tracker.result(), "perplexity": self.pt_perplexity_tracker.result()}

    def test_step(self, inputs):
        # input_sequences, target_sequences = inputs
        # input_sequences, target_sequences, piece_types = inputs
        model_inputs, model_labels, cross_inputs, is_white = inputs
        m_inputs = [model_inputs, cross_inputs, is_white]
        predictions, val_predcitions = self(m_inputs, training=False)
        bloss = self.pt_loss_fn(model_labels, predictions)

        # DISTRIBUTED TRAINING
        if config.distributed is True:
            loss = tf.nn.compute_average_loss(bloss, global_batch_size=config.global_batch_size)
        else:
            loss = bloss

        self.pt_loss_tracker.update_state(loss)
        self.pt_perplexity_tracker.update_state(model_labels, predictions)

        return {"loss": self.pt_loss_tracker.result(), "perplexity": self.pt_perplexity_tracker.result()}

    @property
    def metrics(self):
        return [self.pt_loss_tracker, self.pt_perplexity_tracker]





if __name__ == '__main__':
    model = ChessGPTa2()

    # model_input = ['[white] d2d4 e2e4 a2a4']
    # opp_input = ['[black] d7d5 e7e5 a7a5']
    # is_white = [True]
    #
    # model_input_tensor = config.encode_tf(model_input)
    # opp_input_tensor = config.encode_tf(opp_input)
    # is_white = tf.convert_to_tensor(is_white)
    # inputs = [model_input_tensor, opp_input_tensor, is_white]
    #
    # output = model(inputs)
    # print(output)



    model_input = ['[black] d7d5 e7e5 a7a5']
    opp_input = ['[white] d2d4 e2e4 a2a4 b2b4']
    is_white = [False]

    model_input_tensor = config.encode_tf(model_input)
    opp_input_tensor = config.encode_tf(opp_input)
    is_white = tf.convert_to_tensor(is_white)
    inputs = [model_input_tensor, opp_input_tensor, is_white]

    output = model(inputs)
    print(output)







