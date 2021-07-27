from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam


class CNNNetworks:

    def create_resnet(learning_rate, loss_function, load_pretrained=False,  top='max'):

        if load_pretrained:
            weights = 'imagenet'
        else:
            weights = None

        if top not in (None, 'avg', 'max'):
            raise ValueError('unexpected top pooling layer type: %s' % top)

        # Shape of the input
        input_image_shape = (128, 128, 3)

        # Import base model
        base_model = ResNet50(input_shape=input_image_shape, include_top=False, weights=weights, pooling=top)

        # Only modify Last layer
        for layer in base_model.layers:
            layer.trainable = False

        # Build Model
        altered_model = base_model.output
        #altered_model = Dropout(.5)(altered_model)
        altered_model = Dense(1, activation='sigmoid')(altered_model)

        regr_model = Model(inputs=base_model.inputs, outputs=altered_model)

        # Compile model with Adam optimizer
        regr_model.compile(optimizer=Adam(lr=learning_rate), loss = loss_function, metrics=['mse', 'mae', 'mape'])

        return regr_model

    def create_mobilenet(learning_rate, loss_function, load_pretrained=False, top='max'):

        if load_pretrained:
            weights = 'imagenet'
        else:
            weights = None

        if top not in (None, 'avg', 'max'):
            raise ValueError('unexpected top pooling layer type: %s' % top)

        # Shape of the input
        input_image_shape = (128, 128, 3)

        # Import base model
        base_model = MobileNetV2(input_shape=input_image_shape, include_top=False, weights=weights, pooling=top)

        # Only modify Last layer
        for layer in base_model.layers:
            layer.trainable = False

        # Build Model
        altered_model = base_model.output
        altered_model = Dense(1, activation='sigmoid')(altered_model)

        regr_model = Model(inputs=base_model.inputs, outputs=altered_model)

        # Compile model with Adam optimizer
        regr_model.compile(optimizer=Adam(lr=learning_rate), loss = loss_function, metrics=['mse', 'mae', 'mape'])

        return regr_model
