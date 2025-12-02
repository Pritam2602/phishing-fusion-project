import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve


parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true", help="Force CPU only")
parser.add_argument("--model", default="models/audio_cnn_best.h5", help="Path to trained model")
parser.add_argument("--test-csv", default="data/audio/splits/test.csv", help="Path to test CSV")
parser.add_argument("--out-dir", default="outputs/audio/", help="Output directory")
args = parser.parse_args()

if args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.makedirs(args.out_dir, exist_ok=True)


N_MELS = 64
MEL_WIDTH = 256


def load_mel(path):
    p = None
    try:
        if hasattr(path, "numpy"):
            p = path.numpy()
    except:
        p = None

    if isinstance(p, (bytes, bytearray)):
        p = p.decode("utf-8")
    elif p is None:
        if isinstance(path, (bytes, bytearray)):
            p = path.decode("utf-8")
        else:
            p = str(path)

    p = os.path.normpath(p)

    mel = np.load(p)
    mel = mel.astype(np.float32)

    if mel.ndim != 2:
        mel = np.zeros((N_MELS, MEL_WIDTH), dtype=np.float32)

    h, w = mel.shape
    if w < MEL_WIDTH:
        pad = np.zeros((N_MELS, MEL_WIDTH - w), dtype=np.float32)
        mel = np.hstack([mel, pad])
    else:
        mel = mel[:, :MEL_WIDTH]

    return mel


def tf_load_mel(path, label):
    mel = tf.py_function(load_mel, [path], tf.float32)
    mel = tf.expand_dims(mel, -1)               # → (64,256,1)
    mel.set_shape((N_MELS, MEL_WIDTH, 1))
    label = tf.cast(label, tf.float32)
    return mel, label



def build_dataset(csv_path):
    df = pd.read_csv(csv_path)
    label_map = {"benign": 0, "phishing": 1}
    labels = np.array([label_map[l] for l in df["label"]], dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, labels))
    ds = ds.map(tf_load_mel, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(32).prefetch(tf.data.AUTOTUNE)


print("\nLoading model:", args.model)
model = tf.keras.models.load_model(args.model)

print("📥 Loading test dataset:", args.test_csv)
test_ds = build_dataset(args.test_csv)


print("\n🔍 Running evaluation...")
y_true = []
y_pred_prob = []

for batch_x, batch_y in test_ds:
    preds = model.predict(batch_x, verbose=0)
    y_pred_prob.extend(preds.flatten().tolist())
    y_true.extend(batch_y.numpy().flatten().tolist())

y_true = np.array(y_true)
y_pred_prob = np.array(y_pred_prob)
y_pred = (y_pred_prob >= 0.5).astype(int)



report = classification_report(y_true, y_pred, target_names=["benign", "phishing"])
print("\n===============================")
print("CLASSIFICATION REPORT")
print("===============================")
print(report)

# Save report
with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
    f.write(report)


cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6,5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0,1], ["Benign", "Phishing"])
plt.yticks([0,1], ["Benign", "Phishing"])
plt.savefig(os.path.join(args.out_dir, "confusion_matrix.png"))
plt.close()


fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(args.out_dir, "roc_curve.png"))
plt.close()

prec, rec, _ = precision_recall_curve(y_true, y_pred_prob)
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig(os.path.join(args.out_dir, "pr_curve.png"))
plt.close()

print(f"\n📁 All evaluation outputs saved in: {args.out_dir}\n")
