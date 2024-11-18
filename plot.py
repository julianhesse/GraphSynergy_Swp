import matplotlib.pyplot as plt

# البيانات التي لديك (مع Cross-Validation)
epochs = [1, 2, 3, 4, 5, 6]
train_loss_cv = [0.76, 0.59, 0.55, 0.53, 0.51, 0.50]
val_loss_cv = [0.62, 0.58, 0.56, 0.54, 0.54, 0.53]
train_accuracy_cv = [0.62, 0.70, 0.73, 0.75, 0.76, 0.76]
val_accuracy_cv = [0.68, 0.71, 0.73, 0.73, 0.74, 0.74]
train_roc_auc_cv = [0.66, 0.77, 0.81, 0.83, 0.84, 0.84]
val_roc_auc_cv = [0.74, 0.78, 0.80, 0.81, 0.82, 0.82]

# البيانات التي لديك (بدون Cross-Validation)
epochs_no_cv = [1, 2,3,4,5,6]
train_loss_no_cv = [0.78, 0.60, 0.56, 0.53, 0.52,0.50]
val_loss_no_cv = [0.63, 0.59,0.58,0.56,0.56,0.56]
train_accuracy_no_cv = [0.61, 0.70,0.73,0.74,0.75,0.76]
val_accuracy_no_cv = [0.67, 0.70,0.72,0.73,0.73,0.73]
train_roc_auc_no_cv = [0.64, 0.77,0.8,0.82,0.83,0.84]
val_roc_auc_no_cv = [0.73, 0.77,0.79,0.8,0.8,0.81]

# رسم منحنيات الخسارة
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_loss_cv, label='Training Loss (CV)', marker='o')
plt.plot(epochs, val_loss_cv, label='Validation Loss (CV)', marker='o')
plt.plot(epochs_no_cv, train_loss_no_cv, label='Training Loss (No CV)', marker='x')
plt.plot(epochs_no_cv, val_loss_no_cv, label='Validation Loss (No CV)', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss (With and Without Cross-Validation)')
plt.legend()
plt.grid(True)
plt.show()

# رسم منحنيات الدقة
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_accuracy_cv, label='Training Accuracy (CV)', marker='o')
plt.plot(epochs, val_accuracy_cv, label='Validation Accuracy (CV)', marker='o')
plt.plot(epochs_no_cv, train_accuracy_no_cv, label='Training Accuracy (No CV)', marker='x')
plt.plot(epochs_no_cv, val_accuracy_no_cv, label='Validation Accuracy (No CV)', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs. Validation Accuracy (With and Without Cross-Validation)')
plt.legend()
plt.grid(True)
plt.show()

# رسم منحنيات ROC-AUC
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_roc_auc_cv, label='Training ROC AUC (CV)', marker='o')
plt.plot(epochs, val_roc_auc_cv, label='Validation ROC AUC (CV)', marker='o')
plt.plot(epochs_no_cv, train_roc_auc_no_cv, label='Training ROC AUC (No CV)', marker='x')
plt.plot(epochs_no_cv, val_roc_auc_no_cv, label='Validation ROC AUC (No CV)', marker='x')
plt.xlabel('Epoch')
plt.ylabel('ROC AUC')
plt.title('Training vs. Validation ROC AUC (With and Without Cross-Validation)')
plt.legend()
plt.grid(True)
plt.show()
