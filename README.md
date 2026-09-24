# Rugby Stats LNR

Application Streamlit d’exploration des statistiques de joueurs de rugby. Elle utilise `lnr_stats_flat.sqlite` et les visuels du dossier `images/` présents dans ce dépôt.

## Lancement local

Python 3.11 ou 3.12 recommandé.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Mise en ligne avec Streamlit Community Cloud

1. Connecter son compte GitHub sur [share.streamlit.io](https://share.streamlit.io/).
2. Cliquer sur **Create app**, puis sélectionner ce dépôt, la branche `main` (après intégration de la PR) et `app.py` comme fichier principal.
3. Choisir l’URL souhaitée et cliquer sur **Deploy**. Les dépendances sont installées depuis `requirements.txt` ; la base SQLite et les images sont lues directement depuis le dépôt.

Le dépôt est actuellement privé : l’accès à l’application et les possibilités de déploiement dépendent du compte et des réglages de partage Streamlit. Les données incluses dans le dépôt peuvent être exposées par une application publique ; vérifier les droits de diffusion avant de choisir cette visibilité.

Pour actualiser les chiffres, mettre à jour `lnr_stats_flat.sqlite` dans le dépôt, puis redémarrer ou redéployer l’application. La base est lue en consultation par l’app.
