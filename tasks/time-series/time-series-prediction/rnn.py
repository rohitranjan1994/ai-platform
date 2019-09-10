from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from nn import *
from keras.optimizers import SGD
import data_prep
from keras.models import model_from_json
from itertools import product
from functools import partial, update_wrapper
import time
from keras import backend as K
from keras.regularizers import L1L2


class RNN:
    def __init__(self, input_shape, output_dim):
        self.input_length, self.input_dim = input_shape[0], input_shape[1]
        self.output_dim = output_dim
        self.model = self.__prepare_model()

    def __prepare_model(self):
        print('Build model...')

        model = Sequential()
        model.add(LSTM(64, return_sequences=True,
                       input_shape=(self.input_length, self.input_dim),bias_regularizer=L1L2(l1=0.01, l2=0.01)))
        model.add(LSTM(64, return_sequences=False, input_shape=(self.input_length, self.input_dim)))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(self.output_dim, activation='softmax'))
        #model.add(bias_regularizer=L1L2(l1=0.01, l2=0.01))

        print('Compile model.   ..')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def get_model(self):
        return self.model


def main():

    timestep = 50
    n_cross_validation = 3
    # for order book info only
    data = data_prep.get_test_data(timestep, predict_step=5, filename="./data/order_book_sample.csv")

    # input_shape <- (timestep, n_features)
    # output_shape <- n_classes
    nn = NeuralNetwork(RNN(input_shape=data.x.shape[1:], output_dim=data.y.shape[1]), class_weight= {0:1., 1:1., 2:1.})

    print("TRAIN")
    nn.train(data)

    print("TEST")
    nn.test(data)

    # print("TRAIN WITH CROSS-VALIDATION")
    # nn.run_with_cross_validation(data, n_cross_validation)



if __name__ == '__main__':
    main()
