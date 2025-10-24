import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.models import Model
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- A. CUSTOM F1-SCORE METRIC
class F1Score(keras.metrics.Metric):
    def __init__(self, name='f1_score', num_classes=None, **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.argmax(y_true, axis=-1)
        y_pred = K.argmax(y_pred, axis=-1)
        
        y_true_flat = K.flatten(y_true)
        y_pred_flat = K.flatten(y_pred)

        cm = tf.math.confusion_matrix(y_true_flat, y_pred_flat, num_classes=self.num_classes)
        
        TP = tf.linalg.diag_part(cm)
        FP = tf.reduce_sum(cm, axis=0) - TP
        FN = tf.reduce_sum(cm, axis=1) - TP
        
        self.true_positives.assign_add(tf.reduce_sum(TP))
        self.false_positives.assign_add(tf.reduce_sum(FP))
        self.false_negatives.assign_add(tf.reduce_sum(FN))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1

    def reset_state(self):
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)




DATA_DIR = "/kaggle/input/plantdoc-s/archive/train" 
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
TOTAL_EPOCHS_PHASE_1 = 10
TOTAL_EPOCHS_PHASE_2 = 40


W_P_FINAL = 1.0; W_D_FINAL = 1.0; W_P_TEMP = 0.5; W_D_TEMP = 0.5 
INIT_LR_PHASE_1 = 1e-3
INIT_LR_PHASE_2 = 1e-5


def map_labels(class_names):
    plant_names = []
    disease_names = set()

    for full_name in class_names:
        # Kiểm tra nếu tên lớp 
        if '_' not in full_name:
            print(f"⚠️ Cảnh báo: Tên lớp không hợp lệ (thiếu '_'), bỏ qua: {full_name}")
            continue
            
        # Tách tên Cây (plant) ở dấu gạch dưới đầu tiên
        
        parts = full_name.split('_', 1)
        
        # Đảm bảo có cả Plant và Disease
        if len(parts) < 2:
             print(f"⚠️ Cảnh báo: Tên lớp không hợp lệ (không có Disease), bỏ qua: {full_name}")
             continue
             
        plant = parts[0]
        disease = parts[1]

        # 1. Tách Plant Name
        if plant not in plant_names:
            plant_names.append(plant)
        
        # 2. Gộp Bệnh
        if 'healthy' in disease.lower(): # Sử dụng .lower() để đảm bảo
            disease_names.add('Healthy')
        else:
            # Gộp các bệnh trùng tên
            disease_names.add(disease)
            
    
    plant_map = {name: i for i, name in enumerate(sorted(plant_names))}
    disease_map = {name: i for i, name in enumerate(sorted(list(disease_names)))}
    
    if not plant_map or not disease_map:
        raise ValueError("Lỗi nghiêm trọng: Không thể trích xuất lớp Cây hoặc Bệnh hợp lệ nào. Hãy đảm bảo thư mục data chứa các lớp theo định dạng 'Cây_Bệnh'.")
        
    return plant_map, disease_map


class MultiOutputDataGenerator(keras.utils.Sequence):
    def __init__(self, datagen, plant_map, disease_map, batch_size=32, shuffle=True):
        self.generator = datagen
        self.plant_map = plant_map
        self.disease_map = disease_map
        self.batch_size = batch_size
        self.num_plants = len(plant_map)
        self.num_diseases = len(disease_map)
        self.n = self.generator.n
        self.on_epoch_end()

    def __len__(self):
        return self.generator.__len__()

    def __getitem__(self, index):
        x_batch, _ = self.generator.__getitem__(index)
        
        
        file_indices = self.generator.index_array[index * self.batch_size: (index + 1) * self.batch_size]
        class_names = [self.generator.filenames[i].split(os.sep)[0] for i in file_indices]
        
        plant_labels = []
        disease_labels = []
        
        for full_name in class_names:
            
            
            if '_' not in full_name:
                raise ValueError(f"Malformed class name found in batch: {full_name}. Check your dataset directory names.")
            
            parts = full_name.split('_', 1)
            if len(parts) < 2:
                 raise ValueError(f"Malformed class name found in batch (No disease part): {full_name}.")

            plant = parts[0]
            disease = parts[1]
           
            
            
            plant_labels.append(self.plant_map[plant])
            
            if 'healthy' in disease.lower():
                disease_labels.append(self.disease_map['Healthy'])
            else:
                disease_labels.append(self.disease_map[disease])

        
        Y_plant = tf.one_hot(np.array(plant_labels), self.num_plants)
        Y_disease = tf.one_hot(np.array(disease_labels), self.num_diseases)

        
        return x_batch, {
            "plant_output": Y_plant, "disease_output": Y_disease,
            "plant_output_t": Y_plant, "disease_output_t": Y_disease
        }
    
    def on_epoch_end(self):
        self.generator.on_epoch_end()


def create_cross_fusion_model(numPlants, numDis, input_shape):
    base_model = keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_shape=input_shape
    )
    base_model.trainable = False 
    
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ], name="data_augmentation")

    input_layer = keras.Input(shape=input_shape)
    x = data_augmentation(input_layer) 
    x = base_model(x, training=False) # training=False để giữ BN stats
    x = layers.GlobalAveragePooling2D()(x)

   
    output_plant_t = layers.Dense(numPlants, activation="softmax", name="plant_output_t")(x)
    output_dis_t = layers.Dense(numDis, activation="softmax", name="disease_output_t")(x)

    
    d_plant = layers.concatenate([x, output_dis_t], name="fusion_to_plant") 
    d_dis = layers.concatenate([x, output_plant_t], name="fusion_to_disease")

   
    output_plant = layers.Dense(numPlants, activation="softmax", name="plant_output")(d_plant)
    output_dis = layers.Dense(numDis, activation="softmax", name="disease_output")(d_dis)
    
    model = Model(
        inputs=input_layer, 
        outputs=[output_plant, output_dis, output_plant_t, output_dis_t], 
        name='Cross_Fusion_Multi_Class'
    )
    
    return model


def lr_schedule_cosine(epoch, initial_lr):
    total_decay_epochs = TOTAL_EPOCHS_PHASE_2
    current_decay_epoch = epoch - TOTAL_EPOCHS_PHASE_1
    if current_decay_epoch < 0:
        return initial_lr
    
    cosine_decay = 0.5 * (1 + np.cos(np.pi * current_decay_epoch / total_decay_epochs))
    return INIT_LR_PHASE_2 * cosine_decay
datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2 
)

train_generator_base = datagen.flow_from_directory(
    DATA_DIR,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator_base = datagen.flow_from_directory(
    DATA_DIR,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_names = list(train_generator_base.class_indices.keys())
plant_map, disease_map = map_labels(class_names)

NUM_CLASSES_PLANT = len(plant_map)
NUM_CLASSES_DISEASE = len(disease_map)

print(f"Final Plant Classes: {NUM_CLASSES_PLANT}")
print(f"Final Disease Classes (Gộp Khỏe mạnh): {NUM_CLASSES_DISEASE}")


train_generator = MultiOutputDataGenerator(train_generator_base, plant_map, disease_map, BATCH_SIZE)
val_generator = MultiOutputDataGenerator(val_generator_base, plant_map, disease_map, BATCH_SIZE, shuffle=False)


model = create_cross_fusion_model(NUM_CLASSES_PLANT, NUM_CLASSES_DISEASE, INPUT_SHAPE)
BASE_MODEL_NAME = model.get_layer(index=2).name

losses = {
    "plant_output": "categorical_crossentropy", "disease_output": "categorical_crossentropy", 
    "plant_output_t": "categorical_crossentropy", "disease_output_t": "categorical_crossentropy",
}
lossWeights = {
    "plant_output": W_P_FINAL, "disease_output": W_D_FINAL,
    "plant_output_t": W_P_TEMP, "disease_output_t": W_D_TEMP
}
metrics_dict = {
    "plant_output": ['accuracy', F1Score(name='f1', num_classes=NUM_CLASSES_PLANT)],
    "disease_output": ['accuracy', F1Score(name='f1', num_classes=NUM_CLASSES_DISEASE)]
}

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)


model.get_layer(BASE_MODEL_NAME).trainable = True
model.get_layer('data_augmentation').trainable = True

lr_scheduler = keras.callbacks.LearningRateScheduler(
    lambda epoch: lr_schedule_cosine(epoch, INIT_LR_PHASE_2), 
    verbose=0
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=INIT_LR_PHASE_2),
    loss=losses, loss_weights=lossWeights,
    metrics=metrics_dict
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=train_generator.__len__(),
    validation_steps=val_generator.__len__(),
    epochs=TOTAL_EPOCHS_PHASE_1 + TOTAL_EPOCHS_PHASE_2,
    initial_epoch=TOTAL_EPOCHS_PHASE_1,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

