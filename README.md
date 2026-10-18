# 🛡️ NovaPay AI — Prototype FinTech : paiement sécurisé & détection de fraude par IA

> **Proof of Concept** d'une application bancaire intelligente : chaque transaction est scorée
> en temps réel par un moteur d'IA hybride (scikit-learn). Au-delà de **30 % de risque**,
> l'authentification dynamique (**code à usage unique `OTP-XXXX`**) est déclenchée automatiquement.
> Interface **multi-devises** (EUR, USD, MAD, GBP) et **module de simulation de paiement** intégrés.

| Composant | Technologie |
|---|---|
| Backend / API | **FastAPI** + Uvicorn |
| IA / Machine Learning | **scikit-learn** — `RandomForestClassifier` + `IsolationForest` + règles métier |
| Frontend | **HTML + Tailwind CSS + Chart.js** (interface React-like, sans build, responsive, multi-devises) |
| Base de données | **SQLite** (fichier `novapay_demo.db`, créé automatiquement) |

---

## 1. Aperçu

NovaPay AI simule un compte bancaire complet, enrichi d'un **moteur de détection de fraude** :

- **Chaque paiement est analysé** (montant, heure, localisation, type de marchand, appareil, vélocité…).
- **L'IA produit un score de risque de 0 à 100 %** avec une explication en langage naturel.
- **Risque < 30 %** → paiement validé automatiquement.
- **Risque ≥ 30 %** → **authentification dynamique** : un code à usage unique `OTP-XXXX` est généré et doit être validé.
- **3 codes erronés** → la transaction est **bloquée définitivement** (protection anti-fraude).
- **Paiement validé** → un **code de sécurité à usage unique généré par l'IA** est affiché dans le reçu.
- **Multi-devises** : un sélecteur dans l'en-tête (€ EUR, $ USD, MAD, £ GBP) convertit dynamiquement
  tous les montants affichés (solde, revenus, dépenses, transactions, alertes).
- Le **tableau de bord** affiche le solde, les KPI, les graphiques d'analyse comportementale,
  les anomalies détectées et le **score de sécurité dynamique** du client.
- La carte **« Alertes » est cliquable** : elle ouvre l'historique des 3 dernières tentatives de
  fraude (heure, montant, motif du blocage).

---

## 2. Arborescence du projet

```
PE&SI/
├── main.py               # Serveur FastAPI : routes REST, base SQLite, pipeline de décision
├── ml_model.py           # Entraînement + prédiction du risque IA (RandomForest + IsolationForest)
├── index.html            # Interface utilisateur (Tailwind CSS + Chart.js, JS vanilla)
├── requirements.txt      # Dépendances Python
├── run.bat               # Lancement automatique (Windows)
├── run.sh                # Lancement automatique (macOS / Linux)
├── artifacts/            # Modèles IA sérialisés (générés au 1er lancement, auto-rechargés ensuite)
│   ├── random_forest.joblib
│   ├── isolation_forest.joblib
│   └── metrics.json
└── novapay_demo.db       # Base SQLite de démonstration (créée au 1er lancement)
```

---

## 3. Démarrage rapide

### Pré-requis
- **Python 3.10 ou supérieur** (testé avec Python 3.15 sur Windows)
- Connexion Internet au premier chargement de la page (Tailwind CSS & Chart.js sont servis par CDN)

### Option A — Script automatique

**Windows :**
```bat
run.bat
```
**macOS / Linux / Git Bash :**
```bash
chmod +x run.sh && ./run.sh
```

### Option B — Installation manuelle

```bash
# 1) Environnement virtuel
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2) Dépendances
pip install -r requirements.txt

# 3) (Optionnel) Entraîner le modèle IA + auto-évaluation
python ml_model.py --train --selftest

# 4) Lancer le serveur
uvicorn main:app --reload --port 8000
```

### Accès

| URL | Description |
|---|---|
| http://127.0.0.1:8000 | 📊 Interface utilisateur (dashboard + simulateur) |
| http://127.0.0.1:8000/docs | 📚 Documentation interactive Swagger (tester l'API en direct) |
| http://127.0.0.1:8000/api/v1/health | ❤️ Sonde de santé du service |

> Le premier démarrage entraîne le modèle (~10 s) et génère ~45 jours d'historique bancaire
> fictif pour 4 utilisateurs. Les lancements suivants sont instantanés.

---

## 4. Scénario de démonstration (5 minutes)

1. **Simulation libre** → gros bouton **« Simuler un paiement »** (colonne de droite).
   La modale permet de saisir **montant, marchand et pays** (+ catégorie, heure, appareil).
   Pendant la saisie, le **score de risque est calculé en temps réel** (barre animée + facteurs détectés).

2. **Paiement normal** → saisissez un petit montant, un marchand local, votre pays
   (ou bouton « Scénario normal »)
   → score ~5 %, *validé automatiquement*, et un **reçu s'affiche avec le code de sécurité
   à usage unique généré par l'IA** (ex. `OTP-8492`), le solde est débité dans la seconde.

3. **Paiement suspect** → montant ×10, crypto/cartes cadeaux, 3 h du matin, pays à risque,
   appareil inconnu (ou bouton « Scénario suspect »)
   → score > 70 %, **la modale MFA s'ouvre** avec le « SMS simulé » contenant le code `OTP-XXXX`.

4. **Sécurité** :
   - cliquez « Remplir » puis « Valider » → paiement autorisé, **reçu avec le code** ;
   - avec un mauvais code → refus, compteur de tentatives ;
   - au bout de 3 échecs → transaction **bloquée** ;
   - « Renvoyer » génère un nouveau code ; « Refuser » annule sans débit.

5. **Historique des alertes** → cliquez sur la carte **« Alertes »** (KPI en haut à droite) :
   la fenêtre affiche les **3 dernières tentatives de fraude** avec **heure, montant et motif du blocage**.

6. **Multi-devises** → changez la devise dans l'en-tête (`€ EUR`, `$ USD`, `MAD`, `£ GBP`) :
   solde, KPI, graphiques, transactions et alertes sont **reconvertis instantanément**
   (montants stockés en EUR, puis convertis selon le taux affiché).

7. **Impact sur le compte** : après validation, le solde, l'historique, le graphique
   « Analyse comportementale — 14 jours » et le **score de sécurité** se recalculent en direct
   (le score baisse après une transaction à risque).

8. **Mode démo automatique** (interrupteur en bas à droite) : un flux de paiements
   (70 % normaux / 30 % suspects) est envoyé toutes les 7 s ; les codes MFA sont validés
   automatiquement pour illustrer la chaîne complète.

9. **Réinitialiser l'historique** : bouton en bas de la colonne de droite.

---

## 5. API REST

| Méthode | Route | Description |
|---|---|---|
| `POST` | `/api/v1/transaction` | **Cœur du système** : score le risque, déclenche le MFA si ≥ 30 %, et délivre le **code d'autorisation** si validé |
| `POST` | `/api/v1/transaction/preview` | **Aperçu temps réel** du score de risque pendant la saisie (aucune donnée enregistrée) |
| `POST` | `/api/v1/transaction/{id}/verify` | Valide le code MFA `OTP-XXXX` (3 essais max, durée de vie 180 s, anti-rejeu) |
| `POST` | `/api/v1/transaction/{id}/resend` | Renvoie un nouveau code (simulation SMS) |
| `POST` | `/api/v1/transaction/{id}/cancel` | Refus de l'utilisateur → aucun débit |
| `GET` | `/api/v1/transactions?user_id=&only_flagged=` | Historique des transactions (paramètre `currency`) |
| `GET` | `/api/v1/alerts/{user_id}?limit=3` | **Historique des alertes** : 3 dernières tentatives de fraude (heure, montant, motif) |
| `GET` | `/api/v1/dashboard/{user_id}` | Solde, KPI, graphiques, anomalies, score de sécurité, infos IA (paramètre `currency`) |
| `GET` | `/api/v1/analytics/{user_id}` | Vue analytique (catégories, heures, tendance, distribution du risque) |
| `GET` | `/api/v1/currencies` | Devises supportées + taux de change (EUR base) |
| `GET` | `/api/v1/users` | Liste des clients de démonstration |
| `GET` | `/api/v1/merchants` | Référentiels : catégories de marchands, marchands, villes et pays (risques associés) |
| `GET` | `/api/v1/model/insights` | Métriques du modèle, importances des variables, seuils |
| `POST` | `/api/v1/simulate` | Génère un paiement `normal` ou `suspicious` et l'injecte dans le pipeline |
| `POST` | `/api/v1/users/{id}/reset` | Réinitialise l'historique + le solde d'un client |

> 💱 **Multi-devises** : tous les montants sont **stockés en EUR** puis convertis à la volée
> dans la devise demandée (`?currency=USD|MAD|GBP`, taux fixes de démonstration :
> 1 € = 1,09 $ = 10,85 MAD = 0,84 £). La réponse contient `currency`, `rate` et `amount_eur`.
> Le champ `currency` est aussi accepté en écriture : un paiement saisi en USD est converti
> en EUR pour le scoring, puis reconverti pour l'affichage.

### Exemple — paiement suspect (curl)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/transaction \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user_001","amount":2400,"merchant":"Binance",
       "merchant_type":"crypto","country":"RU","hour":3,"is_new_device":true,
       "currency":"EUR"}'
```

```jsonc
{
  "risk_score": 73.1,
  "risk_level": "eleve",
  "decision": "CHALLENGE",
  "requires_otp": true,
  "currency": "EUR",
  "factors": [
    "Montant 52,2x supérieur à votre moyenne habituelle (2400,00 EUR)",
    "Transaction effectuée à 3h (plage horaire sensible 23h-5h)",
    "Localisation inhabituelle : Moscou, RU (2486 km de Paris, FR)",
    "Transaction internationale (hors pays de résidence)",
    "Catégorie marchand à risque élevé : Crypto-actifs",
    "Appareil non reconnu / nouvelle empreinte digitale absente"
  ],
  "authorization_code": null,
  "otp_challenge": { "demo_otp": "OTP-8492", "expires_in_seconds": 180, "max_attempts": 3 }
}
```

Après validation (`POST /api/v1/transaction/{id}/verify` avec `{"otp": "OTP-8492"}`) :

```jsonc
{
  "status": "approved",
  "decision": "ACCEPTED_AFTER_OTP",
  "authorization_code": "OTP-8492",   // le code MFA validé devient le code d'autorisation
  "amount": 2400.0, "currency": "EUR",
  "balance_after": 1785.75
}
```

> ⚠️ **PoC** : sans passerelle SMS, le code MFA est renvoyé dans le champ `demo_otp` et affiché
> dans l'interface (encadré « Simulation SMS »). En production, il serait envoyé par SMS/push
> et **jamais** exposé par l'API.

---

## 6. Comment fonctionne l'IA ?

### Variables analysées (12 features)

`montant`, `écart au montant habituel`, `heure`, `nuit (23h-5h)`, `week-end`,
`localisation inhabituelle`, `distance du domicile (formule de Haversine)`,
`risque de la catégorie marchand`, `vélocité (transactions / heure)`,
`appareil inconnu`, `transaction internationale`, `historique de risque du client`.

### Le score final (0–100 %)

```
score = 100 × ( 0,46 × P(RandomForest)        # classification supervisée
              + 0,34 × P(IsolationForest)     # détection d'anomalies
              + 0,20 × Score de règles )^1,22 # explicabilité & garde-fous
```

- **RandomForestClassifier** (240 arbres) — apprend sur 12 000 transactions synthétiques
  étiquetées (dont des cas « limites » pour éviter un modèle irréaliste).
- **IsolationForest** — entraîné **uniquement sur les transactions légitimes** : tout
  comportement s'écartant de la norme devient une anomalie, y compris une fraude jamais vue.
- **Règles métier** — garantissent des explications lisibles et bornent les décisions.

### Performances obtenues (`python ml_model.py --selftest`)

| Métrique | Valeur |
|---|---|
| AUC-ROC | **1.000** (données synthétiques séparables) |
| Précision / Rappel / F1 | **0.969 / 0.999 / 0.984** |
| Score moyen — légitimes | **7.3 / 100** (p95 = 18.3) |
| Score moyen — fraudes | **92.0 / 100** (p05 = 81.4) |
| Faux positifs au seuil 30 % | **1.37 %** |
| Fraudes détectées au seuil 30 % | **100 %** |

### Seuils de décision

| Score | Niveau | Décision |
|---|---|---|
| `< 30 %` | faible | ✅ Validation automatique + **code d'autorisation à usage unique** |
| `30 – 59 %` | modéré | 🔐 Authentification dynamique (code `OTP-XXXX`) |
| `60 – 84 %` | élevé | 🔐 Code `OTP-XXXX` + signalement dans le dashboard |
| `≥ 85 %` | critique | 🔐 Code `OTP-XXXX` (blocage automatique activable, voir §7) |

---

## 7. Personnalisation

Variables d'environnement (avant de lancer le serveur) :

| Variable | Défaut | Effet |
|---|---|---|
| `OTP_THRESHOLD` | `30` | Seuil de bascule vers l'authentification dynamique |
| `OTP_TTL_SECONDS` | `180` | Durée de vie d'un code OTP |
| `AUTO_BLOCK_RISK_THRESHOLD` | `101` (jamais) | Score au-delà duquel la transaction est refusée sans OTP |
| `NOVAPAY_DB` | `novapay_demo.db` | Chemin du fichier SQLite |

Pour réentraîner le modèle : `python ml_model.py --train --samples 20000` puis supprimer
le dossier `artifacts/` pour forcer la régénération.

**Devises** : les taux de change de démonstration sont définis dans le dictionnaire `CURRENCIES`
de `main.py` (base EUR). Pour brancher des taux réels : appelez une API de change au démarrage
(ou mettez en cache un taux journalier) et alimentez `CURRENCIES[code]["rate"]` — le reste de
l'application (dashboard, alertes, simulateur, scoring) s'adapte automatiquement.

---

## 8. Limites du PoC & pistes d'industrialisation

- **Données d'entraînement synthétiques** → en production : historique réel (anonymisé),
  réentraînement périodique, gestion du *concept drift*, modèle par segment client.
- **SQLite + verrous applicatifs** → PostgreSQL / Kafka pour l'ingestion temps réel.
- **OTP exposé pour la démo** → passerelle SMS/Twilio, TOTP (RFC 6238), WebAuthn / biométrie.
- **Pas d'authentification utilisateur** sur l'API → OAuth2 / mTLS, journal d'audit,
  conformité **DSP2 / RGPD / ACPR** (explicabilité et contestation des décisions automatisées).
- **Explicabilité** → SHAP values pour une justification par transaction opposable au client.

---

## 9. Dépannage

| Problème | Solution |
|---|---|
| `ModuleNotFoundError` | Activez le venv puis `pip install -r requirements.txt` |
| Erreur d'installation de `pyyaml` / compilation C++ | Installez `uvicorn` **sans** l'extra `[standard]` (déjà le cas dans `requirements.txt`) |
| Page sans style (Tailwind) | Vérifiez l'accès Internet (les CDN servent Tailwind et Chart.js) |
| `Artifact déjà existant mais incompatible` | Supprimez le dossier `artifacts/` puis relancez |
| Historique incohérent pendant la démo | `POST /api/v1/users/user_001/reset` ou le bouton « Réinitialiser » |

---

*Prototype à but de démonstration — toutes les données sont fictives.*
