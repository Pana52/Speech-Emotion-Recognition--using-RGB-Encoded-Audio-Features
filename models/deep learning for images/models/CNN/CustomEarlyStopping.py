from keras.callbacks import Callback
import numpy as np

class CustomEarlyStopping(Callback):
    def __init__(self, switch_epoch, min_delta=0, patience=0):
        super().__init__()
        self.switch_epoch = switch_epoch  # Epoch to switch from val_loss to val_accuracy
        self.min_delta = min_delta
        self.patience = patience
        self.best_weights = None  # Store the best weights

        # For monitoring val_loss
        self.best_val_loss = np.Inf
        self.wait_loss = 0  # Counter for patience mechanism for loss

        # For monitoring val_accuracy
        self.best_val_acc = 0
        self.wait_acc = 0  # Counter for patience mechanism for accuracy

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy')

        # Before the switching epoch, monitor val_loss
        if epoch < self.switch_epoch:
            if np.less(val_loss - self.min_delta, self.best_val_loss):
                self.best_val_loss = val_loss
                self.wait_loss = 0
                self.best_weights = self.model.get_weights()
            else:
                self.wait_loss += 1
                if self.wait_loss >= self.patience:
                    print(f"\nEpoch {epoch+1}: early stopping (minimizing val_loss)")
                    self.model.stop_training = True
                    self.model.set_weights(self.best_weights)

        # After the switching epoch, monitor val_accuracy
        else:
            if np.greater(val_acc - self.min_delta, self.best_val_acc):
                self.best_val_acc = val_acc
                self.wait_acc = 0
                self.best_weights = self.model.get_weights()
            else:
                self.wait_acc += 1
                if self.wait_acc >= self.patience:
                    print(f"\nEpoch {epoch+1}: early stopping (maximizing val_accuracy)")
                    self.model.stop_training = True
                    self.model.set_weights(self.best_weights)

# Usage example:
# custom_early_stopping = CustomEarlyStopping(switch_epoch=20, min_delta=0.001, patience=10)
