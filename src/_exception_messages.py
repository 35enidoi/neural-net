from enum import StrEnum


class NNExceptionMessages(StrEnum):
    NN_INIT_LAYER_NOM = "At least an input layer and an output layer are required."
    NN_ACV_LAY_NOM = "The length of activation functions must match the number of layers minus one (excluding the input layer)"
    NN_ACV_FUNC_NOM = "Activation function must be a subclass of AbstractActivationAlgorithm."
    NN_ACV_FUNC_INSTANCE_NOM = "Activation function must be a subclass of AbstractActivationAlgorithmNoStatic."
    NN_PREDICT_INPUT_NOM = "The number of inputs must match the number of input nodes."
