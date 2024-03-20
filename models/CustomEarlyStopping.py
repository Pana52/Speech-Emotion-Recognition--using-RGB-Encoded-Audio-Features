from keras.callbacks import Callback
import numpy as np


class CustomEarlyStoppingAndSaveBest(Callback):
    def __init__(self, monitor_loss='val_loss', monitor_acc='val_accuracy', patience=0, verbose=0):
        super(CustomEarlyStoppingAndSaveBest, self).__init__()
        self.monitor_loss = monitor_loss
        self.monitor_acc = monitor_acc
        self.patience = patience
        self.best_weights = None
        self.best_epoch = 0
        self.epochs_since_last_improvement = 0
        self.verbose = verbose
        self.best_loss = np.Inf
        self.best_acc = 0

    def on_train_begin(self, logs=None):
        # Reset the state of the callback
        self.best_weights = None
        self.best_epoch = 0
        self.epochs_since_last_improvement = 0
        self.best_loss = np.Inf
        self.best_acc = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor_loss)
        current_acc = logs.get(self.monitor_acc)
        if current_loss < self.best_loss or (current_loss == self.best_loss and current_acc > self.best_acc):
            self.best_loss = current_loss
            self.best_acc = current_acc
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            self.epochs_since_last_improvement = 0
        else:
            self.epochs_since_last_improvement += 1
        if self.epochs_since_last_improvement > self.patience:
            self.model.stop_training = True
            if self.verbose > 0:
                print(f'\nEarly stopping at epoch {epoch + 1}')
                print(f'Restoring model weights from the end of the best epoch: {self.best_epoch + 1}.')
            self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        # Ensure the weights of the model are set to the best found during training
        if self.best_weights:
            self.model.set_weights(self.best_weights)
