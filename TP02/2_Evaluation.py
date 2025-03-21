# **************************************************************************
# INF7370 Apprentissage automatique 
# Travail pratique 2 
# ===========================================================================

#===========================================================================
# Dans ce script, on évalue le modèle entrainé dans 1_Modele.py
# On charge le modèle en mémoire; on charge les images; et puis on applique le modèle sur les images afin de prédire les classes



# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

from tensorflow import keras

# La libraire responsable du chargement des données dans la mémoire
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Affichage des graphes
import matplotlib.pyplot as plt

# La librairie numpy 
import numpy as np

# Configuration du GPU
import tensorflow as tf
from keras import backend as K

# Utilisé pour le calcul des métriques de validation
from sklearn.metrics import confusion_matrix, roc_curve , auc

# Utlilisé pour charger le modèle
from keras.models import load_model
from keras import Model

# ==========================================
# ==================MODÈLE==================
# ==========================================

#Chargement du modéle sauvegardé dans la section 1 via 1_Modele.py
model_path = "Model.keras"
Classifier: Model = load_model(model_path)

# ==========================================
# ================VARIABLES=================
# ==========================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       QUESTIONS
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 1) A ajuster les variables suivantes selon votre problème:
# - mainDataPath         
# - number_images        
# - number_images_class_x
# - image_scale          
# - images_color_mode    
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# L'emplacement des images de test
mainDataPath = "mnist/"
testPath = mainDataPath + "test"

# Le nombre des images de test à évaluer
number_images = 2000 # 1000 images pour la classe du chiffre 2 et 1000 pour la classe du chiffre 7
number_images_class_0 = 1000
number_images_class_1 = 1000

# La taille des images à classer
image_scale = 28

# La couleur des images à classer
images_color_mode = "grayscale"  # grayscale or rgb

# ==========================================
# =========CHARGEMENT DES IMAGES============
# ==========================================

# Chargement des images de test
test_data_generator = ImageDataGenerator(rescale=1. / 255)

test_itr = test_data_generator.flow_from_directory(
    testPath,# place des images
    target_size=(image_scale, image_scale), # taille des images
    class_mode="binary",# Type de classification
    shuffle=False,# pas besoin de les boulverser
    batch_size=1,# on classe les images une à la fois
    color_mode=images_color_mode)# couleur des images

(x, y_true) = next(test_itr)

# ==========================================
# ===============ÉVALUATION=================
# ==========================================

# Les classes correctes des images (1000 pour chaque classe) -- the ground truth
y_true = np.array([0] * number_images_class_0 + 
                  [1] * number_images_class_1)

# evaluation du modËle
test_eval = Classifier.evaluate(test_itr, verbose=1)

# Affichage des valeurs de perte et de precision
print('>Test loss (Erreur):', test_eval[0])
print('>Test précision:', test_eval[1])

# Prédiction des classes des images de test
predicted_classes = Classifier.predict(test_itr, verbose=1)
predicted_classes_perc = np.round(predicted_classes.copy(), 4)
predicted_classes = np.round(predicted_classes) # on arrondie le output
# 0 => classe 2
# 1 => classe 7

# Cette list contient les images bien classées
correct = []
for i in range(0, len(predicted_classes) ):
    if predicted_classes[i] == y_true[i]:
        correct.append(i)

# Nombre d'images bien classées
print("> %d  Ètiquettes bien classÈes" % len(correct))

# Cette list contient les images mal classées
incorrect = []
for i in range(0, len(predicted_classes) ):
    if predicted_classes[i] != y_true[i]:
        incorrect.append(i)

# Nombre d'images mal classées
print("> %d Ètiquettes mal classÈes" % len(incorrect))

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 1) Afficher la matrice de confusion
# 2) Extraire une image mal-classée pour chaque combinaison d'espèces - Voir l'exemple dans l'énoncé.
# ***********************************************
