# Dashboard_d'analyse_des_indicateurs_nutritionnels_en_France
Dashboard Nutri-Score – Projet Data Science Santé M1
Analyse du Nutri-Score et des nutriments (Open Food Facts)

#Présentation du projet
Ce projet a pour objectif d’analyser la qualité nutritionnelle des produits alimentaires commercialisés en France à partir des données Open Food Facts. Il s’inscrit dans une démarche de data science en santé publique et vise à mieux comprendre la construction du Nutri-Score à partir des nutriments.
Le projet combine analyse exploratoire, visualisation interactive et modélisation prédictive à travers un dashboard accessible en ligne.

#Objectifs
Analyser la répartition du Nutri-Score selon les catégories alimentaires
Étudier la distribution des nutriments clés (sucre, sel, graisses, fibres, protéines)
Prédire le Nutri-Score lorsqu’il est manquant grâce au Machine Learning
Proposer un outil accessible à un public non spécialiste

#Accès à l’application
Le dashboard est disponible en ligne à l’adresse suivante :
https://dashboard-d-analyse-des-indicateurs.onrender.com
⚠️ L’application est hébergée sur Render. Le premier chargement peut prendre jusqu’à 10 minutes. Il est recommandé d’attendre sans fermer la page.

#Structure du dashboard
Le dashboard est composé de trois pages principales, accessibles grâce aux boutons circulaires situés sous le titre :

- Page 1 – Analyse du Nutri-Score
Cette page permet d’analyser la répartition globale du Nutri-Score.
Fonctionnalités :
Sélection multiple des catégories de produits
KPI indiquant le nombre total de produits et les proportions de scores
Graphique en barres représentant la distribution A → E
Les graphiques et indicateurs se mettent automatiquement à jour en fonction des catégories sélectionnées.

- Page 2 – Analyse des nutriments
Cette page permet d’étudier les nutriments individuellement.
Fonctionnalités :
Sélection d’un seul nutriment à la fois (sucre, sel, matières grasses, protéines, fibres, graisses saturées)
Sélection multiple des catégories (18 catégories disponibles)
KPI affichant les moyennes par catégorie
Carte thermique et graphiques dynamiques
Lorsque plusieurs catégories sont sélectionnées, tous les KPI ne sont pas visibles en même temps. Une barre de défilement verticale permet d’accéder aux indicateurs supplémentaires.

- Page 3 – Modèle de prédiction du Nutri-Score
Cette page permet de prédire le Nutri-Score d’un produit.
Fonctionnalités :
Affichage du nombre de produits analysés
Formulaire de saisie des valeurs nutritionnelles
Bouton de prédiction
Seuls les chiffres sont acceptés dans les champs.
Toutes les valeurs doivent être renseignées pour lancer la prédiction.
Si une valeur est manquante, un message d’erreur s’affiche.

#Structure du Git
Le dépôt GitHub regroupe les données, le code et les documents nécessaires pour visualiser et analyser la qualité nutritionnelle des produits alimentaires via notre dashboard. 


- Fichier1_Nutriscore_Categories.csv : Fichier utilisé pour la page 1 du DashBoard ( Nutriscore, catégories).
- Fichier4.csv : Fichier utilisé pour la page 2 du DashBoard. (Nutriscore, catégories, taux des 6 nutriments)
-Fichier4_.csv : Fichier utilisé pour la page 3 du DashBoard. (Nutriscore ( inconnu et connu)
- Fichier_3.csv : Fichier utilisé pour la page 3 du DashBoard. (Nutriscore, taux des 6 nutriments)
- README.md : Fichier contenant la présentation du projet, les instructions et les informations techniques.
- projet_dashbord_render.py : Code principal du dashboard.
- requirements.txt : Liste des bibliothèques Python nécessaires pour exécuter l’application localement.


#Données utilisées
Source : Open Food Facts (produits vendus en France)
Volume initial :
Environ 1,2 million de produits
12 Go de données
Environ 250 variables
Le prétraitement a été réalisé sur Google Colab afin de réduire le volume et créer des fichiers exploitables.

#Prétraitement des données
Les principales étapes sont :
Normalisation du Nutri-Score
Conversion des variables nutritionnelles
Suppression des valeurs aberrantes
Gestion des valeurs manquantes
Catégorisation par mots-clés
Regroupement en macro-catégories

#Modèle de prédiction
Nous avons développé un modèle de régression logistique multi-classes.
Pipeline :
Imputation par médiane
Standardisation
Pondération des classes
Split 80/20
Le modèle atteint une fiabilité d’environ 63 %.

#Installation locale 
Créer un environnement virtuel :
Windows :
python -m venv .venv
.venv\Scripts\activate
Mac/Linux :
python -m venv .venv
source .venv/bin/activate
Installer les dépendances :
pip install dash pandas numpy plotly scikit-learn joblib
Lancer l’application :
python app.py
Accès local :
http://127.0.0.1:8050

#Limites
Données hétérogènes
Catégorisation approximative
Modèle de base
Difficulté pour certaines catégories (huiles)

Équipe
Selma El Fagrouchi
Qais Kazimi
Tiziri Dridi
Tiziri Dridi
Mehdi
Said


