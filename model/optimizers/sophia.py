import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import ops   # Keras-3 back-end neutral math ops

# --------------------------------------------------------------------- #
#                       Sophia-G  (Keras-3 version)                     #
# --------------------------------------------------------------------- #
@keras.utils.register_keras_serializable(package="Custom")
class Sophia(keras.optimizers.Optimizer):
    r"""Sophia-G optimizer  ─  Keras-3 / TF-2.16+ compatible.

    Parameters
    ----------
    learning_rate : float | keras.optimizers.schedules.LearningRateSchedule
    beta_1        : float  (default 0.9)
    beta_2        : float  (default 0.95)
    epsilon       : float  (default 1e-12)
    rho           : float  (default 0.03)   # clipping threshold
    weight_decay  : float  (decoupled, AdamW-style)
    hessian_update_period : int   (k; refresh diag-Hessian every *k* steps)
    scale_by_batch_size   : bool  (multiply G² by B if grads are *means*)
    """

    def __init__(
        self,
        learning_rate=1e-4,
        beta_1=0.9,
        beta_2=0.95,
        epsilon=1e-12,
        rho=0.03,
        weight_decay=0.0,
        *,
        hessian_update_period=1,
        scale_by_batch_size=True,
        name="Sophia",
        **kwargs,
    ):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)


        # Public hyper-parameters are now **plain Python attributes**
        self.learning_rate         = learning_rate
        self.beta_1                = beta_1
        self.beta_2                = beta_2
        self.epsilon               = epsilon
        self.rho                   = rho
        self.weight_decay          = weight_decay
        self.hessian_update_period = int(hessian_update_period)
        self.scale_by_batch_size   = bool(scale_by_batch_size)

    # ----------------------------------------------------------------- #
    #                 slot creation  (called once, lazily)              #
    # ----------------------------------------------------------------- #
    def build(self, var_list):
        for w in var_list:                       # per trainable weight
            # momentum  m_t
            self.add_variable_from_reference(
                reference=w, name="m", initializer="zeros")
            # Hessian diag EMA  h_t
            self.add_variable_from_reference(
                reference=w, name="hessian", initializer="zeros")

    # ----------------------------------------------------------------- #
    #                  one variable, one gradient                       #
    # ----------------------------------------------------------------- #
    def update_step(self, grad, var, *, lr=None):
        # Keras passes None for sparse updates we don't support
        if grad is None:
            return

        lr = lr if lr is not None else self.learning_rate
        beta1, beta2 = self.beta_1, self.beta_2
        rho,  eps    = self.rho, self.epsilon
        wd           = self.weight_decay

        # --- slots ---------------------------------------------------- #
        m = self.get_variable_from_reference(var, "m")
        h = self.get_variable_from_reference(var, "hessian")

        # --- decoupled weight-decay ----------------------------------- #
        if wd:
            grad = grad + wd * var

        # --- momentum :  m ← β₁ m + (1-β₁) g -------------------------- #
        ops.assign(m, beta1 * m + (1. - beta1) * grad)

        # --- parameter update uses *previous* Hessian ------------------ #
        ratio = ops.clip(
            m / (h + eps),  # m / (h + ε)
            -rho, rho
        )
        eta = lr / rho
        ops.assign_sub(var, eta * ratio)

        # --- refresh Hessian diagonal every k steps -------------------- #
        # (*after* using the old one, exactly like the reference impl.)
        if (self.iterations + 1) % self.hessian_update_period == 0:
            g2 = ops.square(grad)
            if self.scale_by_batch_size and grad.shape.rank:
                # multiply by B if grads were averaged (common case)
                B = ops.cast(ops.shape(grad)[0], grad.dtype)
                g2 = B * g2
            new_h = beta2 * h + (1. - beta2) * g2
            ops.assign(h, new_h)

    # ----------------------------------------------------------------- #
    #                 configuration  (for model.save)                   #
    # ----------------------------------------------------------------- #
    def get_config(self):
        return {
            "learning_rate":         self.learning_rate,
            "beta_1":                self.beta_1,
            "beta_2":                self.beta_2,
            "epsilon":               self.epsilon,
            "rho":                   self.rho,
            "weight_decay":          self.weight_decay,
            "hessian_update_period": self.hessian_update_period,
            "scale_by_batch_size":   self.scale_by_batch_size,
            "name":                  self.name,
        }