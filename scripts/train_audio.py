import argparse
import os


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--cpu", action="store_true", help="Force CPU (sets CUDA_VISIBLE_DEVICES=-1)")
parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
parser.add_argument("--train-csv", default="data/audio/splits/train.csv", help="Train CSV path")
parser.add_argument("--val-csv", default="data/audio/splits/val.csv", help="Validation CSV path")
parser.add_argument("--model-out", default="models/audio_cnn_best.h5", help="Path to save the best model")
parser.add_argument("--final-out", default="models/audio_cnn_final.h5", help="Path to save the final model")
args, _ = parser.parse_known_args()

if args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf

N_MELS = 64
MEL_WIDTH = 256
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs

CSV_TRAIN = args.train_csv
CSV_VAL = args.val_csv



def load_mel(path):
    # `path` can be a tf.EagerTensor, bytes, or str depending on how tf passes it.
    # Safely extract a Python string path in all cases.
    p = None
    try:
        # EagerTensor -> numpy() -> bytes or str
        if hasattr(path, 'numpy'):
            p = path.numpy()
    except Exception:
        p = None

    if isinstance(p, (bytes, bytearray)):
        p = p.decode('utf-8')
    elif p is None:
        # fallback if above didn't work: handle raw python bytes/str
        if isinstance(path, (bytes, bytearray)):
            p = path.decode('utf-8')
        else:
            p = str(path)

    # normalize path separators for Windows/Unix
    p = os.path.normpath(p)
    mel = np.load(p)
    
    if mel is None or mel.size == 0:
        return np.zeros((N_MELS, MEL_WIDTH), dtype=np.float32)

    mel = mel.astype(np.float32)

    # Fix shape if needed
    if mel.ndim != 2:
        return np.zeros((N_MELS, MEL_WIDTH), dtype=np.float32)

    h, w = mel.shape
    if w < MEL_WIDTH:
        pad = np.zeros((N_MELS, MEL_WIDTH - w), dtype=np.float32)
        mel = np.hstack([mel, pad])
    else:
        mel = mel[:, :MEL_WIDTH]

    return mel


def tf_load_mel(path, label):
    mel = tf.py_function(load_mel, [path], tf.float32)
    mel = tf.expand_dims(mel, axis=-1)
    mel.set_shape((N_MELS, MEL_WIDTH, 1))
    label = tf.cast(label, tf.float32)
    return mel, label


def build_dataset(csv_path, shuffle=False):
    df = pd.read_csv(csv_path)

    label_map = {"benign": 0, "phishing": 1}
    labels = np.array([label_map[l] for l in df["label"]], dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, labels))
    ds = ds.map(tf_load_mel, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(4096)

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



def build_cnn():
    inp = tf.keras.Input(shape=(N_MELS, MEL_WIDTH, 1))
    x = inp

    x = tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D((2,2))(x)

    x = tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D((2,2))(x)

    x = tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D((2,2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])
    return model


def main():
    print("\n Loading datasets...")

    # compute class weights from train CSV to help with imbalance
    train_df = pd.read_csv(CSV_TRAIN)
    total = len(train_df)
    count_benign = int((train_df["label"] == "benign").sum())
    count_phish = int((train_df["label"] == "phishing").sum())
    # avoid division by zero
    count_benign = max(1, count_benign)
    count_phish = max(1, count_phish)
    class_weight = {
        0: total / (2.0 * count_benign),
        1: total / (2.0 * count_phish),
    }

    train_ds = build_dataset(CSV_TRAIN, shuffle=True)
    val_ds = build_dataset(CSV_VAL)

    print("\n Building CNN model...")
    model = build_cnn()
    model.summary()

    # Callbacks: early stopping, checkpoint, tensorboard
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.final_out), exist_ok=True)
    import datetime
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", "audio", run_id)
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(args.model_out, save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
    ]

    print("\n Training...\n")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb, class_weight=class_weight)

    # save final model
    model.save(args.final_out)
    print(f"\n Training complete! Best model saved to: {args.model_out}, final model saved to: {args.final_out}")

    # persist training history
    hist_df = pd.DataFrame(history.history)
    hist_csv = os.path.join("logs", "audio", f"history_{run_id}.csv")
    os.makedirs(os.path.dirname(hist_csv), exist_ok=True)
    hist_df.to_csv(hist_csv, index=False)
    print(f"Training history saved to: {hist_csv}")


if __name__ == "__main__":
    main()
