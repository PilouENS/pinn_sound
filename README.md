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
- Domaine 1 m × 1 m, source impulsionnelle gaussienne centrée.
- Grille 201 × 201 (dx = dy ≈ 0,005 m) ; pas de temps `dt = 0,95·dx/(c·√2)` avec `c = 343 m/s` ⇒ `T_final = 6 ms`.
- Obstacle unique rectangulaire fixe (24×18 cellules), position aléatoire hors du centre, condition rigide (∂p/∂n ≈ 0 sur ses faces).
- Conditions aux limites externes : Mur ordre 1 absorbant sur les quatre bords.
- Dataset tabulaire `pressure_dataset.npy` : lignes `[sample_id, x, y, t, pression]` + `obstacle_mask_sample{i}.npy` et métadonnées `obstacles_metadata.npy` (bbox obstacle).
- Visualisation temporelle : `Figures/Simulation_exemple.gif` généré par `Simulation/visua_npy.py`.

## Réseau de Neuronnes 
### Architecture du réseau
- MLP fully-connected `FCN`: entrée (x,y,t) → sortie p(x,y,t).
- 5 couches cachées de 128 neurones, activation `tanh`, initialisation Xavier, dernière couche linéaire.

### Entrainement du réseau
- Perte totale `L = w_data·MSE_data + w_phys·MSE_residu + w_bc·MSE_bords` avec `w_data=1`, `w_phys=1e-3`, `w_bc=1` ; résidu calculé par autograd sur l'équation d'onde (∂²p/∂t² − c²Δp).
- Données : échantillonnage aléatoire du dataset (≈8000 points dans le script). Collocation : 10k points dans [0,1]²×[0,T]. Bords : 2k points sur le périmètre.
- Optimisation en deux temps : Adam (lr 1e-3, 5000 époques) puis L-BFGS (500 itérations) ; sauvegarde des poids dans `Réseau/Poids_modèle/pinn_wave_model.pt`.
- Évaluation rapide : `Réseau/testpinn.py` charge le modèle, prédit un champ `nx×nx` à un temps donné, et compare au dataset (MSE + figures `pred_field.png`).
