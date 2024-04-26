import tensorflow as tf
import config



from model.ChessGPT import ChessGPT



def get_pretrain_model(checkpoint_path=None):
    model = ChessGPT()
    test_input = tf.ones((1, config.seq_length))
    model(test_input)
    if checkpoint_path:
        model.load_weights(checkpoint_path).expect_partial()
    model.summary()
    return model


def get_rl_models(checkpoint_actor=None, checkpoint_critic=None):
    test_input = tf.ones((1, config.seq_length))
    actor_model = ChessGPT()
    actor_model(test_input)
    critic_model = ChessGPT()
    critic_model(test_input)
    if checkpoint_actor:
        actor_model.load_weights(checkpoint_actor).expect_partial()
    if checkpoint_critic:
        critic_model.load_weights(checkpoint_critic).expect_partial()
    actor_model.summary()
    return actor_model, critic_model

def get_vs_models(checkpoint_actor=None, checkpoint_actor_2=None, checkpoint_critic=None):
    test_input = tf.ones((1, config.seq_length))
    actor_1_model = ChessGPT()
    actor_1_model(test_input)
    actor_2_model = ChessGPT()
    actor_2_model(test_input)
    critic_model = ChessGPT()
    critic_model(test_input)
    if checkpoint_actor:
        actor_1_model.load_weights(checkpoint_actor).expect_partial()
    if checkpoint_actor_2:
        actor_2_model.load_weights(checkpoint_actor_2).expect_partial()
    if checkpoint_critic:
        critic_model.load_weights(checkpoint_critic).expect_partial()
    actor_1_model.summary()
    return actor_1_model, actor_2_model, critic_model






from model.ChessGPTv2 import ChessGPTv2


def get_pretrain_model_v2(checkpoint_path=None):
    model = ChessGPTv2()
    test_input = tf.ones((1, config.seq_length))
    piece_input = tf.ones((1, config.seq_length))
    model([test_input, piece_input])
    if checkpoint_path:
        model.load_weights(checkpoint_path).expect_partial()
    model.summary()
    return model

def get_rl_models_v2(checkpoint_actor=None, checkpoint_critic=None):
    test_input = tf.ones((1, config.seq_length))
    piece_input = tf.ones((1, config.seq_length))

    actor_model = ChessGPTv2()
    actor_model([test_input, piece_input])

    critic_model = ChessGPTv2()
    critic_model([test_input, piece_input])

    if checkpoint_actor:
        actor_model.load_weights(checkpoint_actor).expect_partial()
    if checkpoint_critic:
        critic_model.load_weights(checkpoint_critic).expect_partial()

    actor_model.summary()
    return actor_model, critic_model





from model.ChessGPTa2 import ChessGPTa2

def get_pretrain_model_a2(checkpoint_path=None):
    model = ChessGPTa2()
    model_input = tf.ones((1, config.seq_length))
    cross_input = tf.ones((1, config.seq_length))
    is_white = tf.convert_to_tensor([True])
    model([model_input, cross_input, is_white])
    if checkpoint_path:
        model.load_weights(checkpoint_path).expect_partial()
    model.summary()
    return model







