# **************************************************************************
# INF7370 Apprentissage automatique
# Travail pratique 3
# ===========================================================================

# ===========================================================================
# Dans ce script, on évalue l'autoencodeur entrainé dans 1_Modele.py sur les données tests.
# On charge le modèle en mémoire puis on charge les images tests en mémoire
# 1) On évalue la qualité des images reconstruites par l'autoencodeur
# 2) On évalue avec une tache de classification la qualité de l'embedding
# 3) On visualise l'embedding en 2 dimensions avec un scatter plot


# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des données dans la mémoire
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Affichage des graphes et des images
import matplotlib.pyplot as plt

# La librairie numpy
import numpy as np

# Configuration du GPU
import tensorflow as tf

# Utlilisé pour charger le modèle
from keras.models import load_model
from keras import Model
from keras.layers import Flatten

# Utilisé pour normaliser l'embedding et faire la classification
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE

from keras import backend as K

# Pour mesurer le temps d'exécution
import time

# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={"GPU": 2, "CPU": 4})
sess = tf.compat.v1.Session(config=config)
try:
    tf.config.experimental.set_memory_growth(
        tf.config.list_physical_devices("GPU")[0], True
    )
except:
    print("No GPU available or GPU memory growth setting failed")

# ==========================================
# ==================MODÈLE==================
# ==========================================

# Chargement du modéle (autoencodeur) sauvegardé dans la section 1
start_time = time.time()
model_path = "Model.keras"
autoencoder = load_model(model_path)
print(f"Modèle chargé en {time.time() - start_time:.2f} secondes")

# Création du modèle d'encodeur en identifiant automatiquement la dernière couche MaxPooling2D
pooling_layers = [
    l for l in autoencoder.layers if isinstance(l, tf.keras.layers.MaxPooling2D)
]
if not pooling_layers:
    # Si pas de MaxPooling2D, on utilise la méthode originale avec l'index fixe
    encoder_layer_index = 9  # À ajuster selon votre architecture
    encoder = Model(
        inputs=autoencoder.input, outputs=autoencoder.layers[encoder_layer_index].output
    )
else:
    # On utilise la dernière couche MaxPooling2D comme sortie de l'encodeur
    encoder = Model(inputs=autoencoder.input, outputs=pooling_layers[-1].output)

print("\nRésumé du modèle d'encodeur:")
encoder.summary()

# ==========================================
# ================VARIABLES=================
# ==========================================

# L'emplacement des images
mainDataPath = "/content/donnees/"

# On évalue le modèle sur les images tests
testPath = mainDataPath + "test"

# Configuration des images
image_scale = 64  # Taille des images (ajustée à 64x64 comme dans le modèle)
image_channels = 3  # Nombre de canaux de couleurs (3 pour RGB)
images_color_mode = "rgb"  # Mode couleur des images

# Nombre d'images à charger
batch_size = 64  # Taille de batch pour le chargement

# ==========================================
# =========CHARGEMENT DES IMAGES============
# ==========================================

start_time = time.time()
# Chargement des images test
test_data_generator = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_data_generator.flow_from_directory(
    testPath,  # Place des images de test
    color_mode=images_color_mode,  # couleur des images
    target_size=(image_scale, image_scale),  # taille des images
    batch_size=batch_size,  # taille du batch
    class_mode="binary",  # On veut les labels pour la classification
    shuffle=False,
)  # Ne pas mélanger pour garder l'ordre des classes

# Calculer le nombre total d'images et par classe
total_images = test_generator.samples
class_names = list(test_generator.class_indices.keys())
print(f"Classes trouvées: {class_names}")
print(f"Nombre total d'images de test: {total_images}")

# Créer les labels pour toutes les images
x_list, y_list = [], []
test_generator.reset()

# Charger toutes les images et leurs labels
for i in range(len(test_generator)):
    batch_x, batch_y = next(test_generator)
    x_list.append(batch_x)
    y_list.append(batch_y)

x_test = np.vstack(x_list)
y_test = np.hstack(y_list)

print(f"Forme des données de test: {x_test.shape}")
print(f"Forme des labels de test: {y_test.shape}")
print(f"Données de test chargées en {time.time() - start_time:.2f} secondes")

# ==========================================
# =======RECONSTRUCTION DES IMAGES==========
# ==========================================

start_time = time.time()
# Reconstruire les images tests en utilisant l'autoencodeur
reconstructed_images = autoencoder.predict(x_test, batch_size=batch_size)
print(f"Reconstruction effectuée en {time.time() - start_time:.2f} secondes")

# Afficher quelques images originales et leurs reconstructions pour chaque classe
plt.figure(figsize=(15, 6))

# Trouver les indices de chaque classe
class_0_indices = np.where(y_test == 0)[0]
class_1_indices = np.where(y_test == 1)[0]

# Afficher 3 exemples pour la classe 0
for i in range(3):
    if i < len(class_0_indices):
        # Image originale
        plt.subplot(2, 6, i + 1)
        plt.imshow(x_test[class_0_indices[i]])
        plt.title(f"Original - Classe: {class_names[0]}")
        plt.axis("off")

        # Image reconstruite
        plt.subplot(2, 6, i + 7)
        plt.imshow(reconstructed_images[class_0_indices[i]])
        plt.title(f"Reconstruite - Classe: {class_names[0]}")
        plt.axis("off")

# Afficher 3 exemples pour la classe 1
for i in range(3):
    if i < len(class_1_indices):
        # Image originale
        plt.subplot(2, 6, i + 4)
        plt.imshow(x_test[class_1_indices[i]])
        plt.title(f"Original - Classe: {class_names[1]}")
        plt.axis("off")

        # Image reconstruite
        plt.subplot(2, 6, i + 10)
        plt.imshow(reconstructed_images[class_1_indices[i]])
        plt.title(f"Reconstruite - Classe: {class_names[1]}")
        plt.axis("off")

plt.tight_layout()
plt.show()

# Afficher un exemple par classe comme dans le code adapté
for i, name in enumerate(class_names):
    idx = np.where(y_test == i)[0][0]
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(x_test[idx])
    plt.title(f"Original - {name}")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_images[idx])
    plt.title(f"Reconstructed - {name}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"reconstructed_{name}.png")
    plt.show()

# ==========================================
# =========EXTRACTION DE L'EMBEDDING========
# ==========================================

start_time = time.time()
# Extraire l'embedding pour les images de test
embeddings = encoder.predict(x_test, batch_size=batch_size)
print(f"Embedding extrait en {time.time() - start_time:.2f} secondes")
print(f"Forme des embeddings avant flatten: {embeddings.shape}")

# Aplatir (flatten) l'embedding pour obtenir un vecteur pour chaque image
embeddings_flat = embeddings.reshape(embeddings.shape[0], -1)
print(f"Forme des embeddings après flatten: {embeddings_flat.shape}")

# ==========================================
# =========NORMALISATION DE L'EMBEDDING=====
# ==========================================

# Normaliser l'embedding avec StandardScaler
scaler = StandardScaler()
embeddings_normalized = scaler.fit_transform(embeddings_flat)
print(f"Forme des embeddings normalisés: {embeddings_normalized.shape}")

# ==========================================
# =======ANALYSE DE LA RECONSTRUCTION=======
# ==========================================

# Calculer l'erreur de reconstruction moyenne par classe
mse_by_class = {}
for i, class_name in enumerate(class_names):
    class_indices = np.where(y_test == i)[0]
    class_mse = np.mean(
        np.square(x_test[class_indices] - reconstructed_images[class_indices])
    )
    mse_by_class[class_name] = class_mse

# Afficher les résultats
print("\nErreur de reconstruction moyenne (MSE) par classe:")
for class_name, mse in mse_by_class.items():
    print(f"Classe {class_name}: {mse:.6f}")

# Calculer l'erreur de reconstruction totale
total_mse = np.mean(np.square(x_test - reconstructed_images))
print(f"Erreur de reconstruction totale (MSE): {total_mse:.6f}")

# ==========================================
# =====CLASSIFICATION SUR L'EMBEDDING=======
# ==========================================

start_time = time.time()
# Grid search pour trouver le meilleur paramètre C pour SVM
param_grid = {"C": [0.1, 1, 10, 100]}
svm = SVC(kernel="linear")
grid = GridSearchCV(svm, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(embeddings_normalized, y_test)
best_C = grid.best_params_["C"]
best_score = grid.best_score_
print(f"Meilleur SVM C: {best_C}, CV accuracy: {best_score:.4f}")
print(f"Grid search effectué en {time.time() - start_time:.2f} secondes")

# SVM avec le meilleur C trouvé
best_svm = SVC(kernel="linear", C=best_C)
y_pred = cross_val_predict(best_svm, embeddings_normalized, y_test, cv=5, n_jobs=-1)
# print(classification_report(y_test, y_pred, target_names=class_names))
# print("Matrice de confusion:")
# print(confusion_matrix(y_test, y_pred))

# Appliquer SVM linéaire sur l'embedding normalisé avec la méthode originale
svm_embedding = SVC(kernel="linear", C=1)
cv_scores_embedding = cross_val_score(
    svm_embedding, embeddings_normalized, y_test, cv=5, scoring="accuracy"
)
print("\nClassification sur l'embedding (SVM linéaire):")
print(
    f"Accuracy moyenne (CV 5-fold): {cv_scores_embedding.mean():.4f} ± {cv_scores_embedding.std():.4f}"
)

# ==========================================
# =====CLASSIFICATION SUR IMAGES ORIGINALES==
# ==========================================

# Aplatir les images originales pour l'entrée dans SVM
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Appliquer SVM linéaire sur les images originales
svm_original = SVC(kernel="linear", C=1)

# Évaluer avec validation croisée (5-fold)
cv_scores_original = cross_val_score(
    svm_original, x_test_flat, y_test, cv=5, scoring="accuracy"
)

print("\nClassification sur images originales (SVM linéaire):")
print(
    f"Accuracy moyenne (CV 5-fold): {cv_scores_original.mean():.4f} ± {cv_scores_original.std():.4f}"
)

# ==========================================
# ========VISUALISATION DE L'EMBEDDING======
# ==========================================

start_time = time.time()
# Réduire la dimensionnalité de l'embedding à 2D avec t-SNE
print("\nApplication de t-SNE pour la visualisation...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings_normalized)
print(f"t-SNE effectué en {time.time() - start_time:.2f} secondes")

# Créer un scatter plot coloré selon les classes
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c=y_test,
    cmap="viridis",
    alpha=0.8,
    edgecolors="w",
    linewidth=0.5,
)

# Ajouter une légende et des titres
plt.colorbar(scatter, label="Classe")
plt.title("Visualisation t-SNE de l'embedding", fontsize=14)
plt.xlabel("Composante 1", fontsize=12)
plt.ylabel("Composante 2", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

# Ajouter une légende explicite pour les classes
for i, class_name in enumerate(class_names):
    plt.scatter([0], [0], c=[i], cmap="viridis", label=f"Classe {i}: {class_name}")
plt.legend()

plt.tight_layout()
plt.show()

# Visualisation par classe comme dans le code adapté
plt.figure(figsize=(8, 6))
for i, name in enumerate(class_names):
    pts = embeddings_2d[y_test == i]
    plt.scatter(pts[:, 0], pts[:, 1], label=name, alpha=0.7)
plt.title("t-SNE des embeddings de test")
plt.xlabel("Composante 1")
plt.ylabel("Composante 2")
plt.legend()
plt.grid(True)
plt.savefig("tsne.png")
plt.show()
