import tensorflow as tf
from typing import Callable, Dict, Union

try:
    # Optional import – TensorFlow Addons provides helpful type hints
    from tensorflow_addons.utils.types import FloatTensorLike  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    # Fall back to a broad numeric type if TFA is not installed.
    FloatTensorLike = Union[float, tf.Tensor]


@tf.keras.utils.register_keras_serializable(package="Addons")
class RectifiedAdam(tf.keras.optimizers.Optimizer):
    """Rectified Adam (RAdam) optimizer **compatible with TensorFlow ≥ 2.19**.

    This is a rewrite of the original TensorFlow‑Addons implementation that
    depended on the deprecated *KerasLegacyOptimizer* interface.  The algorithm
    itself is unchanged – only the integration with Keras' v3 optimizer API has
    been modernised.

    The main differences from the historic API are:
    * Inherit from :class:`tf.keras.optimizers.Optimizer` (v3) instead of the
      v1/v2 *KerasLegacyOptimizer*.
    * Implement :py:meth:`build`, :py:meth:`update_step`, and keep
      hyper‑parameters via :py:meth:`_set_hyper`.
    * Slot variables (`m`, `v`, `vhat`) are created with
      :py:meth:`add_variable_from_reference`.

    Parameters
    ----------
    learning_rate : float | tf.Tensor | tf.keras.optimizers.schedules.LearningRateSchedule
        Base learning‑rate.  Can be a constant or any Keras schedule.
    beta_1, beta_2 : float
        Exponential decay rates for the first and second moment estimations.
    epsilon : float
        Small constant added to denominators to improve numerical stability.
    weight_decay : float | tf.Tensor | schedule, default 0.0
        L2 weight decay.  When > 0, the update is performed *before* applying
        the learning‑rate scale, matching AdamW semantics.
    amsgrad : bool, default False
        Enable the AMSGrad variant.
    sma_threshold : float, default 5.0
        Threshold for the Simple Moving Average (SMA) length below which the
        rectification term is skipped.
    total_steps : int, default 0
        When > 0, enables a linear warm‑up followed by a linear decay to
        `min_lr` as described in the original paper.
    warmup_proportion : float, default 0.1
        Fraction of `total_steps` used for linear warm‑up.
    min_lr : float, default 0.0
        Final learning‑rate after the linear decay phase ends.
    name : str, default "RectifiedAdam"
    """

    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable, Dict] = 1e-3,
        beta_1: FloatTensorLike = 0.9,
        beta_2: FloatTensorLike = 0.999,
        epsilon: FloatTensorLike = 1e-7,
        weight_decay: Union[FloatTensorLike, Callable, Dict] = 0.0,
        amsgrad: bool = False,
        sma_threshold: FloatTensorLike = 5.0,
        total_steps: int = 0,
        warmup_proportion: FloatTensorLike = 0.1,
        min_lr: FloatTensorLike = 0.0,
        name: str = "RectifiedAdam",
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        # Deserialize possible schedule dictionaries (e.g. from a JSON config)
        if isinstance(learning_rate, Dict):
            learning_rate = tf.keras.optimizers.schedules.deserialize(learning_rate)
        if isinstance(weight_decay, Dict):
            weight_decay = tf.keras.optimizers.schedules.deserialize(weight_decay)

        # --- Hyper‑parameters -------------------------------------------------
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("weight_decay", weight_decay)
        self._set_hyper("sma_threshold", sma_threshold)
        self._set_hyper("total_steps", float(total_steps))
        self._set_hyper("warmup_proportion", warmup_proportion)
        self._set_hyper("min_lr", min_lr)

        # --- Attributes ------------------------------------------------------
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self.amsgrad = bool(amsgrad)
        self._initial_total_steps = int(total_steps)
        self._has_weight_decay = weight_decay not in (0.0, 0)

        # These will be populated in `build`.
        self._m: list[tf.Variable] = []
        self._v: list[tf.Variable] = []
        self._vhat: list[tf.Variable] | None = [] if self.amsgrad else None

    # ---------------------------------------------------------------------
    # Keras v3 optimizer API
    # ---------------------------------------------------------------------
    def build(self, var_list):
        """Initialize slot variables after the model variables are known."""
        for var in var_list:
            # First‑moment vector
            self._m.append(
                self.add_variable_from_reference(reference_variable=var, name="m")
            )
            # Second‑moment vector
            self._v.append(
                self.add_variable_from_reference(reference_variable=var, name="v")
            )
            if self.amsgrad:
                # Maintain the maximum second moment for AMSGrad
                assert self._vhat is not None  # mypy
                self._vhat.append(
                    self.add_variable_from_reference(reference_variable=var, name="vhat")
                )

    # pylint: disable=too-many-locals, too-many-branches
    def update_step(self, grad, var, index):
        """Perform a single optimisation step for `var`.

        This method is called for *each* `(grad, var)` pair returned from
        gradient computation.  All hyper‑parameters are *tensorised* so that
        TensorFlow can perform automatic casting and tracing.
        """
        if grad is None:
            return  # pragma: no cover – no gradient for this variable

        dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(dtype)  # Handles learning‑rate schedules

        # Handle weight‑decay schedule if present
        if self._has_weight_decay:
            wd_t = self._decayed_wd(dtype)
        else:
            wd_t = tf.constant(0.0, dtype)

        beta_1 = self._get_hyper("beta_1", dtype)
        beta_2 = self._get_hyper("beta_2", dtype)

        # 1‑indexed step
        step = tf.cast(self.iterations + 1, dtype)
        beta_1_pow = tf.pow(beta_1, step)
        beta_2_pow = tf.pow(beta_2, step)

        m = self._m[index]
        v = self._v[index]

        # m <- beta1 * m + (1 - beta1) * g
        m.assign(beta_1 * m + (1.0 - beta_1) * grad)
        # v <- beta2 * v + (1 - beta2) * g^2
        v.assign(beta_2 * v + (1.0 - beta_2) * tf.square(grad))

        # Bias‑corrected first and second moments
        m_corr = m / (1.0 - beta_1_pow)

        if self.amsgrad:
            assert self._vhat is not None  # for type checker
            vhat = self._vhat[index]
            vhat.assign(tf.maximum(vhat, v))
            v_corr_denom = tf.sqrt(vhat / (1.0 - beta_2_pow))
        else:
            v_corr_denom = tf.sqrt(v / (1.0 - beta_2_pow))

        # --- Rectification terms -----------------------------------------
        sma_inf = 2.0 / (1.0 - beta_2) - 1.0
        sma_t = sma_inf - 2.0 * step * beta_2_pow / (1.0 - beta_2_pow)

        r_t = tf.sqrt(
            (sma_t - 4.0)
            / (sma_inf - 4.0)
            * (sma_t - 2.0)
            / (sma_inf - 2.0)
            * sma_inf
            / sma_t
        )

        sma_threshold = self._get_hyper("sma_threshold", dtype)
        is_rectified = sma_t >= sma_threshold

        # Parameter update before learning‑rate scaling (AdamW‑style)
        if self._has_weight_decay:
            update = m_corr / (v_corr_denom + self.epsilon)  # shape‑like var
            update += wd_t * var
        else:
            update = m_corr / (v_corr_denom + self.epsilon)

        update = tf.where(is_rectified, r_t * update, m_corr)  # RAdam choice

        # ---------------------------------------------------------------
        # Optional warm‑up + linear decay of the *effective* learning rate
        # ---------------------------------------------------------------
        if self._initial_total_steps > 0:
            total_steps = self._get_hyper("total_steps", dtype)
            warmup_steps = total_steps * self._get_hyper("warmup_proportion", dtype)
            min_lr = self._get_hyper("min_lr", dtype)

            decay_steps = tf.maximum(total_steps - warmup_steps, 1.0)
            decay_rate = (min_lr - lr_t) / decay_steps

            lr_t = tf.where(
                step <= warmup_steps,
                lr_t * (step / warmup_steps),  # Linear warm‑up
                lr_t + decay_rate * tf.minimum(step - warmup_steps, decay_steps),
            )

        # ---------------------------------------------------------------
        # Final parameter update
        # ---------------------------------------------------------------
        var.assign_sub(lr_t * update)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _decayed_wd(self, dtype):
        """Return the current weight‑decay, handling schedules transparently."""
        wd = self._get_hyper("weight_decay", dtype)
        if isinstance(wd, tf.keras.optimizers.schedules.LearningRateSchedule):
            wd = tf.cast(wd(self.iterations), dtype)
        return wd

    # ------------------------------------------------------------------
    # Configuration & serialization
    # ------------------------------------------------------------------
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta_1": self._serialize_hyperparameter("beta_1"),
            "beta_2": self._serialize_hyperparameter("beta_2"),
            "epsilon": self.epsilon,
            "weight_decay": self._serialize_hyperparameter("weight_decay"),
            "amsgrad": self.amsgrad,
            "sma_threshold": self._serialize_hyperparameter("sma_threshold"),
            "total_steps": int(self._serialize_hyperparameter("total_steps")),
            "warmup_proportion": self._serialize_hyperparameter("warmup_proportion"),
            "min_lr": self._serialize_hyperparameter("min_lr"),
        }
