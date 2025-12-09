import os
import glob

# Διαγραφή όλων των παλιών μοντέλων
model_dir = r'D:\Diploma\ResNet_TensorFlow_models_ISIC'
models_to_delete = glob.glob(os.path.join(model_dir, 'resnet_model_epoch_*.h5'))

for model_path in models_to_delete:
    os.remove(model_path)

# Διαγραφή του καλύτερου μοντέλου και του τελικού μοντέλου (αν υπάρχουν)
if os.path.exists(os.path.join(model_dir, 'best_resnet_model.keras')):
    os.remove(os.path.join(model_dir, 'best_resnet_model.keras'))

if os.path.exists(os.path.join(model_dir, 'final_resnet_model.h5')):
    os.remove(os.path.join(model_dir, 'final_resnet_model.h5'))

print("Όλα τα παλιά μοντέλα διαγράφηκαν.")

# Διαγραφή των περιεχομένων του log αρχείου val_accuracy_log.txt
log_file = r'D:\Diploma\ResNet_TensorFlow_models_ISIC\val_accuracy_log.txt'
with open(log_file, 'w'):
    pass

# Διαγραφή των περιεχομένων του training_log.csv
csv_log_file = r'D:\Diploma\ResNet_TensorFlow_models_ISIC\training_log.csv'
with open(csv_log_file, 'w'):
    pass

print("Τα αρχεία log διαγράφηκαν.")
