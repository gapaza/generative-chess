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






