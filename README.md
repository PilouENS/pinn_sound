# PINN for acoustic wave equation
Projet du cours IA du M2FESup (ENS Paris-Saclay) : résolution de l'équation d'onde acoustique 2D via un Physics-Informed Neural Network (PINN) entraîné sur des données de simulation en différences finies.

## Aperçu rapide
Aperçu d'une propagation avec obstacle (données de simulation) :

![Simulation exemple](Figures/Simulation_exemple.gif)
L'objectif est d'entraîner un PINN à prédire le champ de pression p(x,y,t) à partir de données simulées, en intégrant les conditions physiques (équation d'onde, conditions aux limites) dans la fonction de perte.

## architecture du git :
- `Simulation/` : génération du dataset (schéma explicite 2D, obstacle rigide, bords absorbants), animation `.gif` depuis un .npy.
- `Réseau/` : modèle PINN (`pinn.py`), script d'entraînement et poids (`Poids_modèle/pinn_wave_model.pt`), évaluation/visualisation (`testpinn.py`).
- `Figures/` : exemples de sorties (`Simulation_exemple.gif`, `pred_field.png`).

## Mise en place du problème
- Domaine 1 m × 1 m, source impulsionnelle gaussienne centrée à t=0.
- Grille 201 × 201 (dx = dy ≈ 0,005 m) 
- Pas de temps `dt = dx/(c·√2)` avec `c = 343 m/s` ça nous fait dt= 10µs. Soit avec 
- Obstacle unique rectangulaire fixe (24×18 cellules), position aléatoire hors du centre, condition rigide (∂p/∂n ≈ 0 sur ses faces)
- Conditions aux limites externes : Mur ordre 1 absorbant sur les quatre bords
- Dataset tabulaire `pressure_dataset.npy` : lignes `[sample_id, x, y, t, pression]` + `obstacle_mask_sample{i}.npy` et métadonnées `obstacles_metadata.npy` (bbox obstacle)
- Visualisation temporelle : `Figures/Simulation_exemple.gif` généré par `Simulation/visua_npy.py`

## Réseau de Neuronnes 
### Architecture du réseau
- MLP fully-connected `FCN`: entrée (x,y,t) → sortie p(x,y,t).
- 5 couches cachées de 128 neurones, activation `tanh`, initialisation Xavier, dernière couche linéaire.

### Entrainement du réseau
- Perte totale `L = w_data·MSE_data + w_phys·MSE_residu + w_bc·MSE_bords` avec `w_data=1`, `w_phys=1e-3`, `w_bc=1` ; résidu calculé par autograd sur l'équation d'onde (∂²p/∂t² − c²Δp).
- Données : échantillonnage aléatoire du dataset (≈8000 points dans le script). Collocation : 10k points dans [0,1]²×[0,T]. Bords : 2k points sur le périmètre.
- Optimisation en deux temps : Adam (lr 1e-3, 5000 époques) puis L-BFGS (500 itérations) ; sauvegarde des poids dans `Réseau/Poids_modèle/pinn_wave_model.pt`.
- Évaluation rapide : `Réseau/testpinn.py` charge le modèle, prédit un champ `nx×nx` à un temps donné, et compare au dataset (MSE + figures `pred_field.png`).

## Problèmes rencontrés 
### Volumes de données
La taille des données (simulation et entrainement) est très importante. 
Avec notre taille de grille on a 201 * 201 points = 40 401 points de l'espace.
Avec notre pas de temps et pour une durée de siumulation de T_max = 6ms, on a 6000µs / 10µs = 600 points temporels.

Par simulation on a donc 40 401 points d'espace * 600 points de temps = 24 240 600 points de données.

Si on part dans un premier temps sur 100 simulations différentes (obstacle positionné aléatoirement), on a 100 simulations * 40 401 points d'espace * 600 points de temps = 2 424 060 000  points de données. On commence à voi rvenir le problème. 

On peut faire une petite approximation : avec 100 simulations dans les conditions décrites ci-dessus, chaque point de données correspond une pression enregistrée en float32 (4 bytes). On a donc 2 424 060 000 points * 4 bytes = 9 696 240 000 bytes = 9,7 GB de données. Ca commence à faire des gros volumes (avec seulement 100 simu pour le cas le plus simple) pour faire tourner sur nos machines. 

De plus dans le peu de temps du projet nous n'avons pas eu le temps d'optimiser le code pour faire tourner sur GPU, ce qui aurait été nécessaire pour faire tourner sur des gros volumes de données.

### Une entrée -> PINN -> très grande sortie
Le PINN prend en entrée le couple (x_obst,y_obst), la position de l'obstacle, et doit prédire le champ de pression p(x,y,t) sur toute la grille (201x201 points d'espace) et pour tous les points temporels (600 points).
On a donc une entrée de dimension 2 (position de l'obstacle) et une sortie de dimension 201x201x600 = 24 240 600. 
Contrairement au cas de l'équation de la chaleur on ne peut pas se placer en régime permanent, et on doit faire la prédiction pour tous les points temporels.



## Idées d'amélioration
### Faire de l'augmentattion de données 
Notre problème de volume de données à générer peut être réduit fortement en utilisant la symétrie du problème. En effet On pourrait imaginer faire un nombr eplus réduit de simulation et "générer" des nouvelles données en appliquant des symétries (ex : symétrie axiale, rotation de 90°) sur les champs de pression obtenus. C'est cette piste qui avait suivi lors d'un TER (de Pierre-Louis) sur le comptage de µ-algues, et qui avait permis de réduire drastiquement le nombre de données à annoter mannuellement pour faire du deep learning.

### Optimiser le code pour faire tourner sur GPU
Il aurait été nécessaire d'optimiser le code pour faire tourner sur GPU, notamment pour la partie d'entrainement du PINN. En effet, avec les volumes de données que nous avons, faire tourner sur CPU est très lent et limite fortement le nombre de simulations que nous pouvons faire et le nombre d'époques d'entrainement du PINN.

### Articles Intéressants
Dans notre recherche de solutions nous sommes tombés sur deux articles qui nous ont paru très intéressants et qui pourraient être des pistes d'amélioration pour notre projet :
- "Room impulse response reconstruction with physics-informed deep learning" (https://pubs.aip.org/asa/jasa/article-abstract/155/2/1048/3261969/Room-impulse-response-reconstruction-with-physics?redirectedFrom=fulltext) qui obtient la figure suivante :
![Room impulse response reconstruction with physics-informed deep learning](Figures/from_article.gif)
Cet article traite un problème similaire au notre (reconstruction d'un champ de pression acoustique à partir de données de simulation) et utilise un PINN pour faire la reconstruction. Leur entrainement sur données de simulation a duré 9h sur un GPU NVIDIA V100 (32GB de VRAM). Ce qui illustre notre problème à faire tourner ça. 

- "PHYSICS-INFORMED DIFFUSION MODELS" (https://arxiv.org/pdf/2403.14404) Lors de notre reflexion on s'est dit finalement que ce qu'on cherché à faire c'était générer une "vidéo" et donc pourquoi ne pas utiliser un modèle de diffusion ? et pour mettre ç aen lien avec notre cours pourquoi ne faire faire un Physic Informed Model de diffusion ? Et de fil en aiguille nous sommes tombés sur cet article. Ce papier propose d’intégrer directement les lois physiques (PDE) dans l’entraînement des modèles de diffusion en ajoutant un terme de perte basé sur le résidu des équations, afin que les échantillons générés respectent la physique. Cette approche permet de générer des solutions réalistes et physiquement cohérentes, tout en améliorant la généralisation et en réduisant fortement les erreurs par rapport aux méthodes purement data-driven. Cela suggère qu’on pourrait entraîner un modèle génératif capable de produire des champs de pression dans une pièce qui respectent directement l’équation d’onde, plutôt que de résoudre la PDE à chaque fois comme avec le PINN.

## Conclusion
Dans le temps imparti, notre projet s’arrête donc ici. Nous avons découvert un aspect du machine learning qui nous intéresse particulièrement, en lien direct avec notre bagage en physique et des outils modernes d’IA. Le fait d’avoir pu choisir un sujet qui nous parle nous motive à poursuivre ce travail en dehors du cadre du cours, en commençant par étudier plus en détail les articles mentionnés.