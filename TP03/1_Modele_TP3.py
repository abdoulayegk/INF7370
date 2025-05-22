# **************************************************************************
# INF7370 Apprentissage automatique
# Travail pratique 3
# ===========================================================================

# #===========================================================================
# Ce modèle est un Autoencodeur Convolutif entrainé sur l'ensemble de données MNIST afin d'encoder et reconstruire les images des chiffres 2 et 7.
# MNIST est une base de données contenant des chiffres entre 0 et 9 Ècrits à la main en noire et blanc de taille 28x28 pixels
# Pour des fins d'illustration, nous avons pris seulement deux chiffres 2 et 7
#
# Données:
# ------------------------------------------------
# entrainement : classe '2': 1 000 images | classe '7': images 1 000 images
# validation   : classe '2':   200 images | classe '7': images   200 images
# test         : classe '2':   200 images | classe '7': images   200 images
# ------------------------------------------------

# >>> Ce code fonctionne sur.
# >>> Vous devez donc intervenir sur ce code afin de l'adapter aux données du TP3.
# >>> À cette fin repérer les section QUESTION et insérer votre code et modification à ces endroits

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des données dans la mémoire
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Le Model à compiler
from keras.models import Model

# Le type d'optimisateur utilisé dans notre modèle (RMSprop, adam, sgd, adaboost ...)
# L'optimisateur ajuste les poids de notre modèle par descente du gradient
# Chaque optimisateur a ses propres paramètres
# Note: Il faut tester plusieurs et ajuster les paramètres afin d'avoir les meilleurs résultats

from tensorflow.keras.optimizers import Adam

# Les types des couches utlilisées dans notre modèle
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Input,
    BatchNormalization,
    UpSampling2D,
    Activation,
    Dropout,
    Flatten,
    Dense,
    LeakyReLU,
)

# Des outils pour suivre et gérer l'entrainement de notre modèle
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configuration du GPU
import tensorflow as tf

# Affichage des graphes
import matplotlib.pyplot as plt

from keras import backend as K

import time

# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={"GPU": 2, "CPU": 4})
sess = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess);
try:
    tf.config.experimental.set_memory_growth(
        tf.config.list_physical_devices("GPU")[0], True
    )
except:
    print("No GPU available or GPU memory growth setting failed")

# ==========================================
# ================VARIABLES=================
# ==========================================

# Le dossier principal qui contient les données
mainDataPath = "/content/donnees/"

# Le dossier contenant les images d'entrainement (sera aussi utilisé pour la validation)
trainPath = mainDataPath + "entrainement/"

# Fraction de données à utiliser pour la validation
validation_split = (
    0.1  # 10% des données d'entrainement seront utilisées pour la validation
)

# Le nom du fichier du modèle à sauvegarder
model_path = "Model.keras"

# Le nombre d'images d'entrainement
training_ds_size = 3240
validation_ds_size = 360

# Configuration des images
image_scale = 64  # Taille des images
image_channels = 3  # Nombre de canaux de couleurs (3 pour RGB)
images_color_mode = "rgb"  # Mode couleur des images
image_shape = (
    image_scale,
    image_scale,
    image_channels,
)  # Forme des images d'entrées (couche d'entrée du réseau)

# Configuration des paramètres d'entrainement
fit_batch_size = 64  # Taille des batchs
fit_epochs = 150  # Nombre d'époques d'entraînement

# ==========================================
# ==================MODÈLE==================
# ==========================================

# Couche d'entrée:
# Cette couche prend comme paramètre la forme des images (image_shape)
input_layer = Input(shape=image_shape)


# Partie d'encodage (qui extrait les features des images et les encode)
def encoder(input):
    x = Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    x = Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    x = Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    encoded = MaxPooling2D((2, 2), padding="same")(x)

    return encoded


# Partie de décodage (qui reconstruit les images à partir de leur embedding ou la sortie de l'encodeur)
def decoder(encoded):
    x = UpSampling2D((2, 2))(encoded)
    x = Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(image_channels, (3, 3), padding="same", kernel_initializer="he_normal")(
        x
    )
    decoded = Activation("sigmoid")(x)
    return decoded


# Déclaration du modèle:
# La sortie de l'encodeur sert comme entrée à la partie decodeur
model = Model(input_layer, decoder(encoder(input_layer)))

# Affichage des paramétres du modèle
# Cette commande affiche un tableau avec les détails du modèle
# (nombre de couches et de paramétres ...)
model.summary()

# Compilation du modèle :
# loss: On définit la fonction de perte (généralement on utilise le MSE pour les autoencodeurs standards)
# optimizer: L'optimisateur utilisé avec ses paramétres (Exemple : optimizer=adam(learning_rate=0.001) )
# metrics: La valeur à afficher durant l'entrainement, metrics=['mse']
# On suit le loss (ou la difference) de l'autoencodeur entre les images d'entrée et les images de sortie
optimizer = Adam(learning_rate=1e-4)
model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])

# ==========================================
# ==========CHARGEMENT DES IMAGES===========
# ==========================================

# Configure data generators with validation split
training_data_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=validation_split,  # Important: Définit la fraction des données pour la validation
)

# training_generator: indique la méthode de chargement des données d'entrainement
training_generator = training_data_generator.flow_from_directory(
    trainPath,  # Place des images d'entrainement
    color_mode=images_color_mode,  # couleur des images
    target_size=(image_scale, image_scale),  # taille des images
    batch_size=fit_batch_size,  # taille du batch
    class_mode="input",
    subset="training",  # Indique que c'est le sous-ensemble d'entrainement
    shuffle=True,  # Mélanger les données
)  # Comme nous somme entrain de reconstruire les images, alors
# la classe de chacune des pixels de sorite est le pixel d'entrée elle même(Input pixel)

# validation_generator: indique la méthode de chargement des données de validation
# (utilise le même dossier mais avec subset="validation")
validation_generator = training_data_generator.flow_from_directory(
    trainPath,  # Même dossier que l'entrainement
    color_mode=images_color_mode,  # couleur des images
    target_size=(image_scale, image_scale),  # taille des images
    batch_size=fit_batch_size,  # taille du batch
    class_mode="input",
    subset="validation",  # Indique que c'est le sous-ensemble de validation
    shuffle=True,  # Mélanger les données
)  # Comme nous somme entrain de reconstruire les images, alors
# la classe de chacune des pixels de sorite est le pixel d'entrée elle même(Input pixel)

# Calculer le nombre d'étapes (steps) nécessaires pour un epoch
steps_per_epoch = training_generator.samples // fit_batch_size
validation_steps = validation_generator.samples // fit_batch_size

# Préparer des batchs pour visualisation plus tard
x_train_batch = next(training_generator)[0]  # Juste pour afficher quelques exemples
x_val_batch = next(validation_generator)[0]  # Juste pour afficher quelques exemples

# ==========================================
# ==============ENTRAINEMENT================
# ==========================================

# Savegarder le modèle avec le minimum loss sur les données de validation (monitor='val_loss')
# Note: on sauvegarde le modèle seulement quand le validation loss (la perte) diminue
# le loss ici est la difference entre les images originales (input) et les images reconstruites (output)
modelcheckpoint = ModelCheckpoint(
    filepath=model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="auto"
)

# Add Early Stopping callback
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

# Add Learning Rate Reducer callback
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

# Add CSV logger
csv_logger = CSVLogger("training_log.csv")

# entrainement du modèle
# On utilise fit_generator car nous utilisons des générateurs de données
start_time = time.time()
autoencoder = model.fit(
    training_generator,
    epochs=fit_epochs,  # nombre d'epochs
    steps_per_epoch=steps_per_epoch,  # nombre d'étapes par epoch
    verbose=1,  # mets cette valeur à 0, si vous voulez ne pas afficher les détails d'entrainement
    callbacks=[
        modelcheckpoint,
        early_stopping,
        reduce_lr,
        csv_logger,
    ],  # les fonctions à appeler à la fin de chaque epoch (dans ce cas modelcheckpoint: qui sauvegarde le modèle)
    validation_data=validation_generator,
    validation_steps=validation_steps,
)  # données de validation
end_time = time.time()

# ==========================================
# ========AFFICHAGE DES RESULTATS===========
# ==========================================

# Affichage du temps d'exécution
execution_time_minutes = (end_time - start_time) / 60
print(f"Temps d'entraînement total: {execution_time_minutes:.2f} minutes")

# Affichage des performances du modèle
print(f"Perte minimale: {min(autoencoder.history['loss']):.6f}")
print(f"Perte de validation minimale: {min(autoencoder.history['val_loss']):.6f}")

# Affichage de la courbe de perte
plt.figure(figsize=(10, 6))
plt.plot(autoencoder.history["loss"], linewidth=2)
plt.plot(autoencoder.history["val_loss"], linewidth=2)
plt.title("Courbe de perte du modèle")
plt.ylabel("Perte (MSE)")
plt.xlabel("Époque")
plt.legend(["Entraînement", "Validation"], loc="upper right")
plt.grid(True)
plt.savefig("courbe_perte.png")
plt.show()
