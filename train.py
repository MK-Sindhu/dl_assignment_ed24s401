import numpy as np
from keras.datasets import mnist
from keras.datasets import fashion_mnist
import wandb
import argparse


# Neural Network Class: feed_forward_NN_final
class feed_forward_NN_final:
    def __init__(
        self,
        layers,
        optimizer,
        learning_rate,
        momentum,
        beta1,
        beta2,
        beta,
        epsilon,
        weight_decay,
        weight_init,
        activation,
        loss
    ):
        self.layers = layers
        self.layer_n = len(layers)
        self.optimizer = optimizer.lower()
        self.lr = learning_rate
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta_rms = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.weight_init = weight_init.lower()
        self.activation = activation.lower()
        self.loss = loss.lower()

        # Initialize Weights & Biases
        self.weights = []
        self.biases = []
        for i in range(self.layer_n - 1):
            if self.weight_init == "xavier":
                # "Xavier" initialization
                w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(1.0 / layers[i])
            else:
                # "random" initialization
                w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        # Initialize extra params if needed
        if self.optimizer in ["momentum", "nag", "rmsprop", "adam", "nadam"]:
            self.v_w = [np.zeros_like(w) for w in self.weights]
            self.v_b = [np.zeros_like(b) for b in self.biases]
        if self.optimizer in ["adam", "nadam"]:
            self.m_w = [np.zeros_like(w) for w in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.t = 0

    # --- Activation Functions ---
    def sigmoid(self, x):
        # Clip to avoid overflow in exp()
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    def activate(self, x):
        if self.activation == "sigmoid":
            return self.sigmoid(x)
        elif self.activation == "tanh":
            return self.tanh(x)
        elif self.activation == "relu":
            return self.relu(x)
        else:
            # default
            return self.sigmoid(x)

    # Derivatives of Activation
    def derivative(self, a):
        if self.activation == "sigmoid":
            return a * (1 - a)
        elif self.activation == "tanh":
            return 1 - a**2
        elif self.activation == "relu":
            return (a > 0).astype(float)
        else:
            return a * (1 - a)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # --- Forward Pass ---
    def forward_pass(self, x):
        self.h = [x]
        # Hidden layers
        for i in range(self.layer_n - 2):
            z = np.dot(self.h[i], self.weights[i]) + self.biases[i]
            act = self.activate(z)
            self.h.append(act)
        # Output layer: softmax
        z_out = np.dot(self.h[-1], self.weights[-1]) + self.biases[-1]
        out = self.softmax(z_out)
        self.h.append(out)
        return self.h

    # --- Backward Pass ---
    def backward_prop(self, y_true):
        m = y_true.shape[0]
        dw = [None] * (self.layer_n - 1)
        db = [None] * (self.layer_n - 1)

        # Output layer delta
        if self.loss == "cross_entropy":
            # Cross-entropy derivative
            delta = self.h[-1] - y_true
        elif self.loss == "mean_squared_error":
            # MSE derivative wrt softmax
            batch_size_sq = len(self.h[-1])
            classes_sq = len(self.h[-1][0])
            delta = np.zeros((batch_size_sq, classes_sq))
            for i in range(batch_size_sq):
                jacobian_softmax = (
                    np.diag(self.h[-1][i]) - np.outer(self.h[-1][i], self.h[-1][i])
                )
                delta[i] = 2 * np.dot(self.h[-1][i] - y_true[i], jacobian_softmax)
        else:
            # default to cross-entropy style
            delta = self.h[-1] - y_true

        # Backprop through layers
        for i in reversed(range(self.layer_n - 1)):
            dw[i] = np.dot(self.h[i].T, delta) / m
            db[i] = np.sum(delta, axis=0, keepdims=True) / m
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.derivative(self.h[i])
        return dw, db

    # --- Parameter Updates (Non-Nesterov) ---
    def _update_params(self, dw, db):
        # Add weight decay
        for i in range(self.layer_n - 1):
            dw[i] += self.weight_decay * self.weights[i]

        if self.optimizer == "sgd":
            for i in range(self.layer_n - 1):
                self.weights[i] -= self.lr * dw[i]
                self.biases[i] -= self.lr * db[i]

        elif self.optimizer == "momentum":
            for i in range(self.layer_n - 1):
                self.v_w[i] = self.momentum * self.v_w[i] + dw[i]
                self.v_b[i] = self.momentum * self.v_b[i] + db[i]
                self.weights[i] -= self.lr * self.v_w[i]
                self.biases[i] -= self.lr * self.v_b[i]

        elif self.optimizer == "rmsprop":
            for i in range(self.layer_n - 1):
                self.v_w[i] = self.beta_rms * self.v_w[i] + (1 - self.beta_rms) * (
                    dw[i] ** 2
                )
                self.v_b[i] = self.beta_rms * self.v_b[i] + (1 - self.beta_rms) * (
                    db[i] ** 2
                )
                self.weights[i] -= self.lr * dw[i] / (np.sqrt(self.v_w[i]) + self.epsilon)
                self.biases[i] -= self.lr * db[i] / (np.sqrt(self.v_b[i]) + self.epsilon)

        elif self.optimizer == "adam":
            self.t += 1
            for i in range(self.layer_n - 1):
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dw[i]
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db[i]
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (dw[i] ** 2)
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db[i] ** 2)

                # Bias correction
                m_w_hat = self.m_w[i] / (1 - self.beta1**self.t)
                m_b_hat = self.m_b[i] / (1 - self.beta1**self.t)
                v_w_hat = self.v_w[i] / (1 - self.beta2**self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta2**self.t)

                self.weights[i] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                self.biases[i] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        elif self.optimizer == "nadam":
            self.t += 1
            for i in range(self.layer_n - 1):
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dw[i]
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db[i]
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (dw[i] ** 2)
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db[i] ** 2)

                # Bias correction
                m_w_hat = self.m_w[i] / (1 - self.beta1 ** (self.t + 1))
                m_b_hat = self.m_b[i] / (1 - self.beta1 ** (self.t + 1))
                v_w_hat = self.v_w[i] / (1 - self.beta2 ** (self.t + 1))
                v_b_hat = self.v_b[i] / (1 - self.beta2 ** (self.t + 1))

                grad_term_w = self.beta1 * m_w_hat + (1 - self.beta1) * dw[i] / (
                    1 - self.beta1 ** (self.t + 1)
                )
                grad_term_b = self.beta1 * m_b_hat + (1 - self.beta1) * db[i] / (
                    1 - self.beta1 ** (self.t + 1)
                )

                self.weights[i] -= self.lr * grad_term_w / (
                    np.sqrt(v_w_hat) + self.epsilon
                )
                self.biases[i] -= self.lr * grad_term_b / (
                    np.sqrt(v_b_hat) + self.epsilon
                )

    # --- Training Step (Includes Nesterov) ---
    def _train_step(self, x_batch, y_batch):
        if self.optimizer == "nag":
            # Look-ahead
            for i in range(self.layer_n - 1):
                self.weights[i] -= self.lr * self.momentum * self.v_w[i]
                self.biases[i] -= self.lr * self.momentum * self.v_b[i]

            self.forward_pass(x_batch)
            out = self.h[-1]

            # L2 norm (for weight decay)
            l2_norm_weights = sum(np.sum(w**2) for w in self.weights)
            l2_norm_params = l2_norm_weights

            # Loss
            if self.loss == "cross_entropy":
                loss = -np.mean(np.sum(y_batch * np.log(out + 1e-10), axis=1)) + (
                    self.weight_decay / 2
                ) * l2_norm_params
            elif self.loss == "mean_squared_error":
                loss = 0.5 * np.mean(np.sum((out - y_batch) ** 2, axis=1))
            else:
                loss = -np.mean(np.sum(y_batch * np.log(out + 1e-10), axis=1)) + (
                    self.weight_decay / 2
                ) * l2_norm_params

            dW, dB = self.backward_prop(y_batch)

            # Weight decay in gradients
            for i in range(self.layer_n - 1):
                dW[i] += self.weight_decay * self.weights[i]

            # Undo look-ahead
            for i in range(self.layer_n - 1):
                self.weights[i] += self.lr * self.momentum * self.v_w[i]
                self.biases[i] += self.lr * self.momentum * self.v_b[i]

            # Update velocity
            for i in range(self.layer_n - 1):
                self.v_w[i] = self.momentum * self.v_w[i] + dW[i]
                self.v_b[i] = self.momentum * self.v_b[i] + dB[i]

            # Final param update
            for i in range(self.layer_n - 1):
                self.weights[i] -= self.lr * self.v_w[i]
                self.biases[i] -= self.lr * self.v_b[i]

            return loss
        else:
            # Normal forward/back
            self.forward_pass(x_batch)
            out = self.h[-1]

            # L2 norm
            l2_norm_weights = sum(np.sum(w**2) for w in self.weights)
            l2_norm_params = l2_norm_weights

            # Loss
            if self.loss == "cross_entropy":
                loss = -np.mean(np.sum(y_batch * np.log(out + 1e-10), axis=1)) + (
                    self.weight_decay / 2
                ) * l2_norm_params
            elif self.loss == "mean_squared_error":
                loss = 0.5 * np.mean(np.sum((out - y_batch) ** 2, axis=1))
            else:
                loss = -np.mean(np.sum(y_batch * np.log(out + 1e-10), axis=1)) + (
                    self.weight_decay / 2
                ) * l2_norm_params

            dW, dB = self.backward_prop(y_batch)
            self._update_params(dW, dB)
            return loss

    # --- Outer Training Loop ---
    def training(self, x_train, y_train, x_val, y_val, epochs, batch_size):
        for ep in range(epochs):
            idx = np.random.permutation(x_train.shape[0])
            x_train_shuff = x_train[idx]
            y_train_shuff = y_train[idx]
            n_batches = len(x_train) // batch_size
            epoch_loss = 0.0

            # Batches
            for b in range(n_batches):
                start = b * batch_size
                end = start + batch_size
                x_batch = x_train_shuff[start:end]
                y_batch = y_train_shuff[start:end]
                loss = self._train_step(x_batch, y_batch)
                epoch_loss += loss

            avg_loss = epoch_loss / n_batches

            # Validation
            preds = self.predict(x_val)
            val_labels = np.argmax(y_val, axis=1)
            val_acc = np.mean(preds == val_labels)

            val_outputs = self.forward_pass(x_val)[-1]
            
            # training
            preds_train = self.predict(x_train)
            train_labels = np.argmax(y_train, axis=1)
            train_acc = np.mean(preds_train == train_labels)

            # Validation loss
            l2_norm_weights = sum(np.sum(w**2) for w in self.weights)
            l2_norm_params = l2_norm_weights

            if self.loss == "cross_entropy":
                val_loss = -np.mean(
                    np.sum(y_val * np.log(val_outputs + 1e-10), axis=1)
                ) + (self.weight_decay / 2) * l2_norm_params
            elif self.loss == "mean_squared_error":
                val_loss = 0.5 * np.mean(np.sum((val_outputs - y_val) ** 2, axis=1))
            else:
                val_loss = -np.mean(
                    np.sum(y_val * np.log(val_outputs + 1e-10), axis=1)
                ) + (self.weight_decay / 2) * l2_norm_params

            # Log to wandb
            wandb.log(
                {
                    "epoch": ep + 1,
                    "training_loss": avg_loss,
                    "validation_accuracy": val_acc,
                    "validation loss": val_loss,
                    "training accuracy": train_acc
                }
            )
            print(
                f"Epoch {ep+1}/{epochs} - loss={avg_loss:.4f}, "
                f"val_acc={val_acc:.4f}, val_loss={val_loss}"
            )

    # --- Prediction ---
    def predict(self, X):
        self.forward_pass(X)
        return np.argmax(self.h[-1], axis=1)


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Neural Network with backpropagation.")

    parser.add_argument("-wp","--wandb_project",type=str,default="q4_sweep_project",help="Project name used to track experiments in Weights & Biases dashboard",)

    # You can omit --wandb_entity or set it explicitly if you have permission:
    # parser.add_argument('-we', '--wandb_entity', type=str, default='myusername', help='Wandb Entity')

    parser.add_argument("-d","--dataset",type=str,default="fashion_mnist",choices=["mnist", "fashion_mnist"],help="Dataset to use",)
    parser.add_argument("-e"  , "--epochs", type=int, default=1, help="Number of epochs to train neural network")
   
    parser.add_argument("-b"  , "--batch_size", type=int, default=4, help="Batch size used to train neural network")
   
    parser.add_argument("-l"  , "--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function",)
   
    parser.add_argument("-o"  , "--optimizer", type=str, default="rmsprop", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Optimizer type",)
   
    parser.add_argument("-lr" , "--learning_rate", type=float, default=0.001, help="Learning rate used to optimize model parameters")
   
    parser.add_argument("-m"  ,"--momentum", type=float, default=0.9, help="Momentum used by momentum and nag optimizers",)
   
    parser.add_argument("-beta", "--beta", type=float, default=0.9, help="Beta used by rmsprop optimizer")
   
    parser.add_argument("-beta1","--beta1",type=float,default=0.9,help="Beta1 used by adam and nadam optimizers",)
   
    parser.add_argument("-beta2","--beta2",type=float,default=0.999,help="Beta2 used by adam and nadam optimizers",)
   
    parser.add_argument("-eps", "--epsilon", type=float, default=0.00000001, help="Epsilon used by optimizers")
   
    parser.add_argument("-w_d","--weight_decay",type=float,default=0.0,help="Weight decay used by optimizers")
   
    parser.add_argument("-w_i","--weight_init",type=str,default="random",choices=["random", "xavier"],help="Weight initialization method",)
   
    parser.add_argument("-nhl","--num_layers",type=int,default=3,help="Number of hidden layers used in feedforward neural network",)
   
    parser.add_argument("-sz","--hidden_size",type=int,default=64,help="Number of hidden neurons in each hidden layer",)
   
    parser.add_argument("-a","--activation",type=str,default="tanh",choices=["identity", "sigmoid", "tanh", "relu"],help="Activation function",)

    # Use parse_known_args to avoid errors with Jupyter’s extra flags
    args, unknown = parser.parse_known_args()
    print(args)
    config = vars(args)

 
    wandb.init(project=config["wandb_project"], config=config, anonymous="allow")
    run_name = "hl_"+str(config["num_layers"])+"_bs_"+str(config["batch_size"])+"_ac_"+str(config["activation"])
    wandb.run.name = run_name
    # Load data
    if config["dataset"] == "fashion_mnist":
        (x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
    else:  # "mnist"
        (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

    x_train_full = x_train_full.reshape(x_train_full.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

    np.random.seed(42)
    idx = np.arange(x_train_full.shape[0])
    np.random.shuffle(idx)
    x_train_full = x_train_full[idx]
    y_train_full = y_train_full[idx]

    # 90% train, 10% validation
    train_size = int(0.9 * len(x_train_full))
    x_train, y_train = x_train_full[:train_size], y_train_full[:train_size]
    x_val, y_val = x_train_full[train_size:], y_train_full[train_size:]

    num_classes = 10
    y_train_1h = np.eye(num_classes)[y_train]
    y_val_1h = np.eye(num_classes)[y_val]
    y_test_1h = np.eye(num_classes)[y_test]

    # Build model
    # layers = [784] + [hidden_size] * num_layers + [10]
    model = feed_forward_NN_final(
        layers=[784] + [config["hidden_size"]] * config["num_layers"] + [10],
        optimizer=config["optimizer"],
        learning_rate=config["learning_rate"],
        momentum=config["momentum"],
        beta1=config["beta1"],
        beta2=config["beta2"],
        beta=config["beta"],
        epsilon=config["epsilon"],
        weight_decay=config["weight_decay"],
        weight_init=config["weight_init"],
        activation=config["activation"],
        loss=config["loss"],
    )

    # Train
    model.training(
        x_train=x_train,
        y_train=y_train_1h,
        x_val=x_val,
        y_val=y_val_1h,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
    )

    # Evaluate on test set
    test_preds = model.predict(x_test)
    test_labels = np.argmax(y_test_1h, axis=1)
    test_acc = np.mean(test_preds == test_labels)

    wandb.log({"test_accuracy": test_acc})
    print("Test accuracy:", test_acc)

    # true_labels = [i in range(10)]

    # ANSWER 7 (Confusion matrix)
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                           y_true=test_labels,
                                                         preds=test_preds,
                                                           class_names=[str(i) for i in range(num_classes)])})