from keras.callbacks import Callback
import numpy as np


class CustomEarlyStoppingAndSaveBest(Callback):
    def __init__(self, monitor_loss='val_loss', monitor_acc='val_accuracy', patience=0, verbose=0, acc_threshold=0.2,
                 start_epoch=5):
        super(CustomEarlyStoppingAndSaveBest, self).__init__()
        self.monitor_loss = monitor_loss
        self.monitor_acc = monitor_acc
        self.patience = patience
        self.start_epoch = start_epoch  # Epoch to start applying the early stopping logic
        self.best_weights = None
        self.best_epoch = 0
        self.epochs_since_last_improvement = 0
        self.verbose = verbose
        self.best_loss = np.Inf
        self.best_acc = 0
        self.acc_threshold = acc_threshold  # Minimum acceptable accuracy

    def on_train_begin(self, logs=None):
        # Initialize best_weights with the model's initial weights
        self.best_weights = self.model.get_weights()
        self.best_epoch = 0
        self.epochs_since_last_improvement = 0
        self.best_loss = np.Inf
        self.best_acc = 0

    def on_epoch_end(self, epoch, logs=None):
        # Only apply early stopping logic after self.start_epoch epochs
        if epoch >= self.start_epoch:
            current_loss = logs.get(self.monitor_loss)
            current_acc = logs.get(self.monitor_acc)
            is_better = False
            if current_acc >= self.acc_threshold:
                if current_loss < self.best_loss:
                    is_better = True
                elif current_loss == self.best_loss and current_acc > self.best_acc:
                    is_better = True

            print(f"Epoch {epoch + 1}, Loss: {current_loss}, Acc: {current_acc}, Best Loss: {self.best_loss}, Best Acc: {self.best_acc}, Improvements since: {self.epochs_since_last_improvement}")

            if is_better:
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
        if self.best_weights:
            self.model.set_weights(self.best_weights)
