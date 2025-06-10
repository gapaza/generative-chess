import tensorflow as tf
import config

"""
        # Model plays white
        # model_moves: [white] d2d4 e2e4 a2a4
        # opp_moves:   [black] d7d5 e7e5 a7a5
        Requires a normal lower triangular mask
        [[
          [True,  False, False, False],
          [True,  True,  False, False],
          [True,  True,  True,  False],
          [True,  True,  True,  True]
        ]]
        


        # Model plays black
        # model_moves: [black] d7d5 e7e5 a7a5
        # opp_moves:   [white] d2d4 e2e4 a2a4
        
            [[

              [True,  True, False, False],
              [True,  True,  True, False],
              [True,  True,  True,  True],
              [True,  True,  True,  True]

            ]]
"""

def generate_batch_of_causal_matrices(conditions):
    """Generate a batch of causal matrices based on a tensor of boolean conditions."""
    def generate_matrix(condition):
        # Choose the matrix type based on the condition
        return tf.cond(
            condition,
            true_fn=lambda: create_white_cross_attn_mask(),
            false_fn=lambda: create_black_cross_attn_mask()
        )

    # Apply the function to each element in the conditions tensor
    batch_matrices = tf.map_fn(generate_matrix, conditions, dtype=tf.bool)
    return batch_matrices

def create_white_cross_attn_mask():
    size = config.seq_length
    white_to_play_cross_mask = tf.linalg.band_part(tf.ones((size, size), dtype=tf.bool), -1, 0)
    return white_to_play_cross_mask

def create_black_cross_attn_mask():
    size = config.seq_length
    lower_triangular = tf.linalg.band_part(tf.ones((size, size), dtype=tf.bool), -1, 0)
    shifted_triangular = tf.roll(lower_triangular, shift=1, axis=1)
    modified_triangular = tf.logical_or(lower_triangular, shifted_triangular)
    black_to_play_cross_mask = tf.tensor_scatter_nd_update(
        modified_triangular,
        indices=[[i, 0] for i in range(size)],
        updates=[True] * size
    )
    return black_to_play_cross_mask









def preprocess_batch(model_inputs, model_labels, opp_inputs, is_white, sample_weights):
    model_inputs_encoded = config.encode_tf(model_inputs)
    model_labels_encoded = config.encode_tf(model_labels)
    opp_inputs_encoded = config.encode_tf(opp_inputs)
    model_labels_encoded = tf.cast(model_labels_encoded, tf.int16)
    model_inputs_encoded = tf.cast(model_inputs_encoded, tf.int16)
    opp_inputs_encoded = tf.cast(opp_inputs_encoded, tf.int16)
    sample_weights = tf.cast(sample_weights, tf.int16)
    return model_inputs_encoded, model_labels_encoded, opp_inputs_encoded, is_white, sample_weights




def preprocess_reward_batch(model_inputs, model_labels, opp_inputs, is_white, rewards, rewards_sample_weights):
    model_inputs_encoded = config.encode_tf(model_inputs)
    model_labels_encoded = config.encode_tf(model_labels)
    opp_inputs_encoded = config.encode_tf(opp_inputs)
    model_labels_encoded = tf.cast(model_labels_encoded, tf.int16)
    model_inputs_encoded = tf.cast(model_inputs_encoded, tf.int16)
    opp_inputs_encoded = tf.cast(opp_inputs_encoded, tf.int16)
    rewards = tf.cast(rewards, tf.float16)
    rewards_sample_weights = tf.cast(rewards_sample_weights, tf.float16)
    return  model_inputs_encoded, model_labels_encoded, opp_inputs_encoded, is_white, rewards, rewards_sample_weights


























if __name__ == '__main__':
    is_white = [True, False]
    is_white = tf.convert_to_tensor(is_white)

    # print(model_input_tensor)
    # print(opp_input_tensor)
    # print(is_white)



    cross_masks = generate_batch_of_causal_matrices(is_white)

    print(cross_masks)

























