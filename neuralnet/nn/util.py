from typing import TypeGuard, Any
from inspect import isclass

from neuralnet.abstracts import (
    AbstractActivationAlgorithm, AbstractActivationAlgorithmNoStatic,
    AbstractLossAlgorithm, AbstractLossAlgorithmNoStatic
)
from neuralnet._exception_messages import NNExceptionMessages


def active_func_check(activate_func: Any) -> TypeGuard[AbstractActivationAlgorithm | AbstractActivationAlgorithmNoStatic]:
    if isclass(activate_func):
        # クラスの時(staticな方じゃないとだめ)
        if not issubclass(activate_func, AbstractActivationAlgorithm):
            raise ValueError(NNExceptionMessages.NN_ACV_FUNC_NOM)
    else:
        # インスタンスの時(どっちでもいい)
        if not issubclass(type(activate_func), (AbstractActivationAlgorithm, AbstractActivationAlgorithmNoStatic)):
            raise ValueError(NNExceptionMessages.NN_ACV_FUNC_INSTANCE_NOM)

    return True


def loss_func_check(loss_func: Any) -> TypeGuard[AbstractLossAlgorithm | AbstractLossAlgorithmNoStatic]:
    if isclass(loss_func):
        if not issubclass(loss_func, AbstractLossAlgorithm):
            return False
    else:
        if not issubclass(type(loss_func), AbstractLossAlgorithm | AbstractLossAlgorithmNoStatic):
            return False
