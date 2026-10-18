"""
ml_model.py
===========
Moteur d'Intelligence Artificielle pour la detection de fraude bancaire.
Prototype FinTech "NovaPay AI".

Le moteur combine 3 signaux complementaires :

  1. RandomForestClassifier  -> apprentissage supervise (donnees synthetiques)
  2. IsolationForest         -> detection d'anomalies non supervisee
  3. Moteur de regles metier -> explicabilite ("pourquoi ce score ?") + garde-fous

Score final = score de risque 0..100 :
  - score <  30  -> risque faible        -> validation automatique (ACCEPT)
  - score >= 30  -> risque modere/eleve  -> authentification dynamique (OTP)

Utilisation CLI :
    python ml_model.py --train      # entraine et sauvegarde le modele
    python ml_model.py --selftest   # auto-evaluation (separation legit / fraude)
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

#: Variables d'entree du modele (dans l'ordre exact attendu par scikit-learn)
FEATURES: List[str] = [
    "amount",
    "amount_ratio",       # montant / montant moyen habituel de l'utilisateur
    "hour",
    "is_night",           # transaction entre 23h et 5h
    "is_weekend",
    "is_new_location",    # localisation differente de la ville de residence
    "distance_km",        # distance par rapport a la ville de residence
    "merchant_risk",      # risque intrinseque de la categorie de marchand
    "tx_velocity_1h",     # nombre de transactions dans la derniere heure
    "is_new_device",
    "is_foreign",         # pays different du pays de residence
    "user_risk_history",  # historique de risque recent de l'utilisateur (0..1)
]

#: Libelles francais des variables (pour l'explicabilite cote client)
FEATURE_LABELS: Dict[str, str] = {
    "amount": "Montant de la transaction",
    "amount_ratio": "Écart vs. montant moyen habituel",
    "hour": "Heure de la transaction",
    "is_night": "Transaction nocturne",
    "is_weekend": "Week-end",
    "is_new_location": "Localisation inhabituelle",
    "distance_km": "Distance / domicile",
    "merchant_risk": "Risque de la catégorie marchand",
    "tx_velocity_1h": "Rythme de transactions",
    "is_new_device": "Appareil non reconnu",
    "is_foreign": "Transaction internationale",
    "user_risk_history": "Historique de risque du client",
}

#: Categories de marchand (risque intrinseque 0..1, plus c'est eleve plus
#: la categorie est utilisee par les fraudeurs : crypto, cartes cadeaux...)
MERCHANT_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "alimentation":   {"label": "Supermarché / Alimentation", "risk": 0.06, "icon": "cart"},
    "restaurant":     {"label": "Restaurants & Cafés",        "risk": 0.08, "icon": "utensils"},
    "transport":      {"label": "Transport & Mobilité",       "risk": 0.10, "icon": "car"},
    "sante":          {"label": "Santé & Pharmacie",          "risk": 0.09, "icon": "heart"},
    "abonnements":    {"label": "Abonnements & Telecom",      "risk": 0.14, "icon": "wifi"},
    "mode":           {"label": "Mode & Retail",              "risk": 0.22, "icon": "shirt"},
    "ecommerce":      {"label": "E-commerce",                 "risk": 0.30, "icon": "box"},
    "voyage":         {"label": "Voyages & Hôtellerie",       "risk": 0.34, "icon": "plane"},
    "electronique":   {"label": "Électronique / High-Tech",   "risk": 0.55, "icon": "laptop"},
    "jeux_argent":    {"label": "Jeux d'argent & Paris",      "risk": 0.74, "icon": "dice"},
    "crypto":         {"label": "Crypto-actifs",              "risk": 0.86, "icon": "coins"},
    "cartes_cadeaux": {"label": "Cartes cadeaux / Recharges", "risk": 0.90, "icon": "gift"},
    "transfert":      {"label": "Transfert P2P",              "risk": 0.42, "icon": "send"},
}

#: Villes connues avec coordonnees GPS (permet de calculer la distance au domicile)
LOCATIONS: Dict[str, Dict[str, Any]] = {
    "Paris, FR":      {"lat": 48.8566, "lon": 2.3522,   "country": "FR", "risk_hint": 0.10},
    "Lyon, FR":       {"lat": 45.7640, "lon": 4.8357,   "country": "FR", "risk_hint": 0.08},
    "Marseille, FR":  {"lat": 43.2965, "lon": 5.3698,   "country": "FR", "risk_hint": 0.12},
    "Lille, FR":      {"lat": 50.6292, "lon": 3.0573,   "country": "FR", "risk_hint": 0.10},
    "Bruxelles, BE":  {"lat": 50.8503, "lon": 4.3517,   "country": "BE", "risk_hint": 0.12},
    "Geneve, CH":     {"lat": 46.2044, "lon": 6.1432,   "country": "CH", "risk_hint": 0.14},
    "Londres, GB":    {"lat": 51.5074, "lon": -0.1278,  "country": "GB", "risk_hint": 0.28},
    "New York, US":   {"lat": 40.7128, "lon": -74.0060, "country": "US", "risk_hint": 0.35},
    "Abidjan, CI":    {"lat": 5.3600,  "lon": -4.0083,  "country": "CI", "risk_hint": 0.45},
    "Dakar, SN":      {"lat": 14.7167, "lon": -17.4677, "country": "SN", "risk_hint": 0.42},
    "Casablanca, MA": {"lat": 33.5731, "lon": -7.5898,  "country": "MA", "risk_hint": 0.40},
    "Lagos, NG":      {"lat": 6.5244,  "lon": 3.3792,   "country": "NG", "risk_hint": 0.62},
    "Dubai, AE":      {"lat": 25.2048, "lon": 55.2708,  "country": "AE", "risk_hint": 0.58},
    "Hong Kong, HK":  {"lat": 22.3193, "lon": 114.1694, "country": "HK", "risk_hint": 0.66},
    "Moscou, RU":     {"lat": 55.7558, "lon": 37.6173,  "country": "RU", "risk_hint": 0.80},
    "Panama, PA":     {"lat": 8.9824,  "lon": -79.5199, "country": "PA", "risk_hint": 0.82},
}

#: Seuils de decision (identiques cote API, exposes pour le frontend)
OTP_THRESHOLD = 30.0
LEVELS: List[Tuple[float, str, str]] = [
    (30.0, "faible", "Risque faible"),
    (60.0, "modere", "Risque modere"),
    (85.0, "eleve", "Risque eleve"),
    (100.1, "critique", "Risque critique"),
]

# Poids de l'ensemble (somme = 1)
W_RANDOM_FOREST = 0.46
W_ISOLATION_FOREST = 0.34
W_RULES = 0.20
#: Courbe de calibration du score final (legerement > 1 = plus discriminant)
CALIBRATION_EXPONENT = 1.22
#: Pente de la sigmoide qui transforme decision_function() de l'IsolationForest
IF_SIGMOID_SCALE = 0.10


# ---------------------------------------------------------------------------
# Utilitaires geographiques
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance orthodromique en km entre deux points GPS."""
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def location_info(name: Optional[str]) -> Dict[str, Any]:
    """Retourne les infos d'une localisation (ou des valeurs neutres)."""
    if name and name in LOCATIONS:
        info = dict(LOCATIONS[name])
        info["name"] = name
        return info
    return {"lat": None, "lon": None, "country": None, "risk_hint": 0.25, "name": name or "Inconnue"}


def distance_between(a: Optional[str], b: Optional[str]) -> float:
    """Distance en km entre deux villes connues (0.0 si inconnue)."""
    ia, ib = location_info(a), location_info(b)
    if None in (ia.get("lat"), ia.get("lon"), ib.get("lat"), ib.get("lon")):
        return 0.0
    return haversine_km(ia["lat"], ia["lon"], ib["lat"], ib["lon"])


def merchant_risk(merchant_type: str) -> float:
    """Risque intrinseque d'une categorie de marchand (0..1)."""
    return float(MERCHANT_CATEGORIES.get(merchant_type, {"risk": 0.35})["risk"])


def merchant_label(merchant_type: str) -> str:
    return str(MERCHANT_CATEGORIES.get(merchant_type, {"label": merchant_type.replace("_", " ").title()})["label"])


# ---------------------------------------------------------------------------
# Moteur
# ---------------------------------------------------------------------------

class FraudEngine:
    """Moteur hybride RandomForest + IsolationForest + regles metier."""

    def __init__(
        self,
        artifact_dir: str = ARTIFACT_DIR,
        auto_train: bool = True,
        verbose: bool = True,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.verbose = verbose
        self.rf: Optional[RandomForestClassifier] = None
        self.iso: Optional[IsolationForest] = None
        self.metrics: Dict[str, Any] = {}
        self.trained_at: Optional[str] = None
        self.n_train_samples: int = 0
        os.makedirs(self.artifact_dir, exist_ok=True)

        if not self._load() and auto_train:
            self.train()
        if (self.rf is None or self.iso is None) and auto_train:
            raise RuntimeError("Le moteur IA n'est pas initialise (entrainement impossible).")

    # ------------------------------------------------------------------ train

    def train(self, n_samples: int = 12000, seed: int = RANDOM_STATE) -> Dict[str, Any]:
        """Genere un jeu de donnees synthetiques, entraine et evalue le modele."""
        if self.verbose:
            print(f"[IA] Generation de {n_samples} transactions synthetiques...")
        df = self._generate_dataset(n_samples=n_samples, seed=seed)

        X = df[FEATURES]
        y = df["is_fraud"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y
        )

        # 1) RandomForest : classification supervisee
        self.rf = RandomForestClassifier(
            n_estimators=240,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
        self.rf.fit(X_train, y_train)

        # 2) IsolationForest : entraine uniquement sur les transactions legitimes
        legit_train = X_train[y_train == 0]
        self.iso = IsolationForest(
            n_estimators=220,
            contamination=0.06,
            max_samples=min(4096, len(legit_train)),
            random_state=seed,
            n_jobs=-1,
        )
        self.iso.fit(legit_train)

        # 3) Evaluation sur le jeu de test
        if self.verbose:
            print("[IA] Evaluation sur le jeu de test...")
        scores = self.score_frame(X_test)
        preds = (scores >= OTP_THRESHOLD).astype(int)
        rf_proba = self.rf.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, preds).tolist()
        self.metrics = {
            "auc_roc": round(float(roc_auc_score(y_test, scores)), 4),
            "auc_roc_random_forest": round(float(roc_auc_score(y_test, rf_proba)), 4),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
            "confusion_matrix": cm,
            "threshold": OTP_THRESHOLD,
            "n_train_samples": int(len(X_train)),
            "n_test_samples": int(len(X_test)),
            "n_samples_total": int(n_samples),
            "mean_score_legit": round(float(scores[y_test == 0].mean()), 1),
            "mean_score_fraud": round(float(scores[y_test == 1].mean()), 1),
            "p95_score_legit": round(float(np.percentile(scores[y_test == 0], 95)), 1),
            "p05_score_fraud": round(float(np.percentile(scores[y_test == 1], 5)), 1),
            "feature_importances": {
                f: round(float(w), 4)
                for f, w in sorted(
                    zip(FEATURES, self.rf.feature_importances_),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
            },
            "weights": {
                "random_forest": W_RANDOM_FOREST,
                "isolation_forest": W_ISOLATION_FOREST,
                "rules": W_RULES,
            },
        }
        self.trained_at = datetime.now().isoformat(timespec="seconds")
        self.n_train_samples = int(len(X_train))
        self._save()

        if self.verbose:
            print(
                f"[IA] Modele entraine | AUC={self.metrics['auc_roc']:.3f} "
                f"| Precision={self.metrics['precision']:.3f} | Rappel={self.metrics['recall']:.3f}"
            )
            print(
                f"[IA] Score moyen legitime={self.metrics['mean_score_legit']} "
                f"vs fraude={self.metrics['mean_score_fraud']}"
            )
        return self.metrics

    def _generate_dataset(self, n_samples: int, seed: int) -> pd.DataFrame:
        """Genere des transactions synthetiques etiquetees (0 = legitime, 1 = fraude).

        Bonus : ~6% des transactions normales ont des comportements "limites"
        (gros achat en ligne, deplacement...), ce qui evite un modele qui
        separe trop facilement et rend le seuil de 30% credible.
        """
        rng = np.random.default_rng(seed)
        rows: List[Dict[str, float]] = []
        low_risk_cats = ["alimentation", "restaurant", "transport", "sante", "abonnements", "mode"]
        high_risk_cats = ["electronique", "jeux_argent", "crypto", "cartes_cadeaux", "voyage", "transfert"]

        for _ in range(n_samples):
            fraud = rng.random() < 0.30
            avg_amount = float(rng.lognormal(3.7, 0.65))  # ~ 15..300 EUR

            if not fraud:
                borderline = rng.random() < 0.06
                amount = avg_amount * float(rng.lognormal(0.0, 0.45))
                hour = int(np.clip(rng.normal(14, 4.5), 0, 23))
                is_night = hour >= 23 or hour <= 5
                is_weekend = int(rng.random() < 0.28)
                new_location = int(rng.random() < (0.22 if borderline else 0.05))
                distance = float(rng.uniform(150, 1500)) if new_location else float(rng.uniform(0, 25))
                category = str(rng.choice(high_risk_cats if borderline else low_risk_cats))
                velocity = int(rng.integers(0, 3 if not borderline else 5))
                new_device = int(rng.random() < (0.30 if borderline else 0.06))
                foreign = int(rng.random() < (0.25 if borderline else 0.03))
                history = float(rng.beta(1.6, 6.0))
                if borderline:
                    amount *= float(rng.uniform(1.0, 2.6))
            else:
                amount = avg_amount * float(rng.lognormal(1.5, 0.65))  # 3x a 15x
                if rng.random() < 0.72:
                    hour = int(rng.choice([0, 1, 2, 3, 4, 23]))
                else:
                    hour = int(rng.integers(0, 24))
                is_night = hour >= 23 or hour <= 5
                is_weekend = int(rng.random() < 0.34)
                new_location = int(rng.random() < 0.78)
                distance = float(rng.uniform(600, 9000)) if new_location else float(rng.uniform(30, 400))
                category = str(rng.choice(high_risk_cats, p=[0.20, 0.17, 0.26, 0.24, 0.08, 0.05]))
                velocity = int(rng.integers(2, 9))
                new_device = int(rng.random() < 0.68)
                foreign = int(rng.random() < 0.72)
                history = float(rng.beta(3.2, 2.6))

            rows.append(
                {
                    "amount": round(amount, 2),
                    "amount_ratio": round(amount / max(avg_amount, 1.0), 3),
                    "hour": hour,
                    "is_night": int(is_night),
                    "is_weekend": is_weekend,
                    "is_new_location": new_location,
                    "distance_km": round(distance, 1),
                    "merchant_risk": merchant_risk(category),
                    "tx_velocity_1h": velocity,
                    "is_new_device": new_device,
                    "is_foreign": foreign,
                    "user_risk_history": round(history, 3),
                    "is_fraud": int(fraud),
                }
            )
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- scoring

    def _rule_score(self, df: pd.DataFrame) -> np.ndarray:
        """Score de regles metier (0..1), sert aussi de base d'explication."""
        r = (
            np.clip(df["amount_ratio"].to_numpy(dtype=float) / 7.0, 0, 1) * 0.34
            + df["is_night"].to_numpy(dtype=float) * 0.12
            + df["is_new_location"].to_numpy(dtype=float) * 0.14
            + np.clip(df["distance_km"].to_numpy(dtype=float) / 5000.0, 0, 1) * 0.12
            + df["merchant_risk"].to_numpy(dtype=float) * 0.16
            + np.clip(df["tx_velocity_1h"].to_numpy(dtype=float) / 6.0, 0, 1) * 0.12
            + df["is_new_device"].to_numpy(dtype=float) * 0.10
            + df["is_foreign"].to_numpy(dtype=float) * 0.08
            + df["user_risk_history"].to_numpy(dtype=float) * 0.06
        )
        return np.clip(r, 0.0, 1.0)

    def _components(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        assert self.rf is not None and self.iso is not None
        p_rf = self.rf.predict_proba(df[FEATURES])[:, 1]
        decision = self.iso.decision_function(df[FEATURES])
        p_iso = 1.0 / (1.0 + np.exp(decision / IF_SIGMOID_SCALE))
        p_rules = self._rule_score(df)
        raw = W_RANDOM_FOREST * p_rf + W_ISOLATION_FOREST * p_iso + W_RULES * p_rules
        return {"random_forest": p_rf, "isolation_forest": p_iso, "rules": p_rules, "raw": raw}

    def score_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Score de risque 0..100 pour un DataFrame de transactions."""
        comp = self._components(df)
        score = 100.0 * np.power(np.clip(comp["raw"], 0.0, 1.0), CALIBRATION_EXPONENT)
        return np.clip(score, 0.0, 100.0)

    # -------------------------------------------------------------- features

    def build_features(self, tx: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Transforme une transaction brute en vecteur de variables pour l'IA.

        tx      : montant, heure, localisation, type de marchand, appareil...
        profile : ville de residence, montant moyen, historique de risque...
        """
        profile = profile or {}
        amount = float(tx.get("amount") or 0.0)
        avg_amount = max(float(profile.get("avg_amount") or 50.0), 1.0)
        home_city = profile.get("home_city")
        loc_name = tx.get("location") or home_city

        hour = tx.get("hour")
        hour = int(hour) if hour is not None else datetime.now().hour

        loc = location_info(loc_name)
        home = location_info(home_city)
        distance = distance_between(loc_name, home_city) if home_city else 0.0

        same_country = True
        if loc.get("country") and home.get("country"):
            same_country = loc["country"] == home["country"]

        is_new_location = 1 if (home_city and loc_name and loc_name != home_city) else 0
        if loc_name is None and home_city is None:
            is_new_location = 0

        return {
            "amount": round(amount, 2),
            "amount_ratio": round(amount / avg_amount, 3),
            "hour": hour,
            "is_night": int(hour >= 23 or hour <= 5),
            "is_weekend": int(tx.get("is_weekend", 0) if tx.get("is_weekend") is not None else 0),
            "is_new_location": is_new_location,
            "distance_km": round(float(distance), 1),
            "merchant_risk": merchant_risk(str(tx.get("merchant_type") or "ecommerce")),
            "tx_velocity_1h": int(tx.get("tx_velocity_1h") or 0),
            "is_new_device": int(bool(tx.get("is_new_device"))),
            "is_foreign": int(not same_country),
            "user_risk_history": round(float(profile.get("risk_history") or 0.1), 3),
        }

    # -------------------------------------------------------------- predict

    def predict(self, tx: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evalue le risque d'une transaction et explique la decision.

        Retourne un dictionnaire pret a etre serialise en JSON par l'API.
        """
        feats = self.build_features(tx, profile)
        df = pd.DataFrame([feats])[FEATURES]
        comp = self._components(df)
        raw = float(np.clip(comp["raw"][0], 0.0, 1.0))
        score = float(np.clip(100.0 * raw ** CALIBRATION_EXPONENT, 0.0, 100.0))
        score = round(score, 1)

        level, level_label = self.level_for(score)
        requires_otp = score >= OTP_THRESHOLD
        factors = self.explain(feats, tx, profile or {})

        return {
            "risk_score": score,
            "risk_level": level,
            "risk_label": level_label,
            "decision": "CHALLENGE" if requires_otp else "ACCEPTED",
            "requires_otp": requires_otp,
            "threshold": OTP_THRESHOLD,
            "factors": factors,
            "components": {
                "random_forest": round(float(comp["random_forest"][0]) * 100, 1),
                "isolation_forest": round(float(comp["isolation_forest"][0]) * 100, 1),
                "business_rules": round(float(comp["rules"][0]) * 100, 1),
            },
            "features": feats,
            "top_features": self.top_features(feats),
        }

    @staticmethod
    def level_for(score: float) -> Tuple[str, str]:
        for limit, level, label in LEVELS:
            if score < limit:
                return level, label
        return "critique", "Risque critique"

    def explain(self, feats: Dict[str, float], tx: Dict[str, Any], profile: Dict[str, Any]) -> List[str]:
        """Explication lisible en francais des principaux facteurs de risque."""
        factors: List[str] = []
        ratio = feats["amount_ratio"]
        loc_name = tx.get("location") or profile.get("home_city") or "inconnue"
        home = profile.get("home_city")

        if ratio >= 2.5:
            ratio_txt = f"{ratio:.1f}".replace(".", ",")
            amount_txt = f"{tx.get('amount', 0):.2f}".replace(".", ",")
            factors.append(
                f"Montant {ratio_txt}x supérieur à votre moyenne habituelle "
                f"({amount_txt} EUR)"
            )
        if feats["is_night"]:
            factors.append(f"Transaction effectuée à {feats['hour']}h (plage horaire sensible 23h-5h)")
        if feats["is_new_location"]:
            factors.append(
                f"Localisation inhabituelle : {loc_name}"
                + (f" ({feats['distance_km']:.0f} km de {home})" if feats["distance_km"] > 50 else "")
            )
        if feats["is_foreign"]:
            factors.append("Transaction internationale (hors pays de résidence)")
        if feats["merchant_risk"] >= 0.5:
            factors.append(f"Catégorie marchand à risque élevé : {merchant_label(str(tx.get('merchant_type') or ''))}")
        if feats["is_new_device"]:
            factors.append("Appareil non reconnu / nouvelle empreinte digitale absente")
        if feats["tx_velocity_1h"] >= 3:
            factors.append(f"{feats['tx_velocity_1h']} transactions en moins d'une heure (velocity check)")
        if feats["user_risk_history"] >= 0.5:
            factors.append("Historique récent de transactions à risque sur ce compte")

        if not factors:
            factors.append("Profil de transaction cohérent avec vos habitudes (montant, lieu, heure, appareil)")
        return factors

    def top_features(self, feats: Dict[str, float], top: int = 5) -> List[Dict[str, Any]]:
        """Contribution approximative des variables au score (explicabilite)."""
        if self.rf is None:
            return []
        importances = dict(zip(FEATURES, self.rf.feature_importances_))
        scales = {
            "amount": 1500.0, "amount_ratio": 6.0, "hour": 24.0, "is_night": 1.0, "is_weekend": 1.0,
            "is_new_location": 1.0, "distance_km": 6000.0, "merchant_risk": 1.0, "tx_velocity_1h": 6.0,
            "is_new_device": 1.0, "is_foreign": 1.0, "user_risk_history": 1.0,
        }
        contribs = []
        for f in FEATURES:
            value = float(feats.get(f, 0.0))
            contribs.append((f, importances.get(f, 0.0) * min(value / scales[f], 1.0)))
        contribs.sort(key=lambda kv: kv[1], reverse=True)
        total = sum(c for _, c in contribs) or 1.0
        return [
            {
                "feature": f,
                "label": FEATURE_LABELS.get(f, f),
                "value": feats.get(f),
                "impact": round(100.0 * c / total, 1),
            }
            for f, c in contribs[:top]
        ]

    def predict_batch(self, txs: List[Dict[str, Any]], profile: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return [self.predict(tx, profile) for tx in txs]

    # ------------------------------------------------------------------- I/O

    def _paths(self) -> Dict[str, str]:
        return {
            "rf": os.path.join(self.artifact_dir, "random_forest.joblib"),
            "iso": os.path.join(self.artifact_dir, "isolation_forest.joblib"),
            "metrics": os.path.join(self.artifact_dir, "metrics.json"),
        }

    def _save(self) -> None:
        p = self._paths()
        joblib.dump(self.rf, p["rf"])
        joblib.dump(self.iso, p["iso"])
        with open(p["metrics"], "w", encoding="utf-8") as fh:
            json.dump(self.metrics, fh, indent=2, ensure_ascii=False)

    def _load(self) -> bool:
        p = self._paths()
        if not all(os.path.exists(path) for path in p.values()):
            return False
        try:
            self.rf = joblib.load(p["rf"])
            self.iso = joblib.load(p["iso"])
            with open(p["metrics"], "r", encoding="utf-8") as fh:
                self.metrics = json.load(fh)
            self.n_train_samples = int(self.metrics.get("n_train_samples", 0))
            if self.verbose:
                print(f"[IA] Modele charge depuis {self.artifact_dir} (AUC={self.metrics.get('auc_roc')})")
            return True
        except Exception as exc:  # artefact corrompu -> on re-entrainera
            print(f"[IA] Impossible de charger les artefacts ({exc}) -> re-entrainement", file=sys.stderr)
            return False

    def info(self) -> Dict[str, Any]:
        return {
            "algorithm": "Ensemble hybride : RandomForest + IsolationForest + regles metier",
            "version": "1.0.0",
            "trained_at": self.trained_at or self.metrics.get("trained_at"),
            "n_features": len(FEATURES),
            "features": FEATURES,
            "feature_labels": FEATURE_LABELS,
            "weights": self.metrics.get("weights", {
                "random_forest": W_RANDOM_FOREST,
                "isolation_forest": W_ISOLATION_FOREST,
                "rules": W_RULES,
            }),
            "threshold": OTP_THRESHOLD,
            "metrics": {k: v for k, v in self.metrics.items() if k != "feature_importances"},
            "feature_importances": self.metrics.get("feature_importances", {}),
        }


# ---------------------------------------------------------------------------
# Auto-evaluation (CLI)
# ---------------------------------------------------------------------------

def _selftest(engine: FraudEngine, n: int = 6000) -> None:
    print("\n=== AUTO-EVALUATION DU MOTEUR IA ===")
    df = engine._generate_dataset(n_samples=n, seed=987)  # jeu frais, autre graine
    scores = engine.score_frame(df)
    legit = scores[df["is_fraud"] == 0]
    fraud = scores[df["is_fraud"] == 1]

    print(f"Transactions legitimes : {len(legit)}")
    print(f"  score moyen={legit.mean():6.1f} | median={np.median(legit):6.1f} | p95={np.percentile(legit, 95):6.1f}")
    print(f"Transactions frauduleuses : {len(fraud)}")
    print(f"  score moyen={fraud.mean():6.1f} | median={np.median(fraud):6.1f} | p05={np.percentile(fraud, 5):6.1f}")

    challenged_legit = float((legit >= OTP_THRESHOLD).mean() * 100)
    caught_fraud = float((fraud >= OTP_THRESHOLD).mean() * 100)
    print(f"\nSeuil OTP = {OTP_THRESHOLD:.0f}%")
    print(f"  Legitimes challengees (faux positifs) : {challenged_legit:5.2f}%")
    print(f"  Fraudes detectees                     : {caught_fraud:5.2f}%")

    examples = [
        {"label": "Cafe a Paris a 14h", "tx": {"amount": 6.5, "hour": 14, "location": "Paris, FR", "merchant_type": "restaurant"},
         "profile": {"avg_amount": 45, "home_city": "Paris, FR", "risk_history": 0.05}},
        {"label": "Courses habituelles", "tx": {"amount": 78.0, "hour": 19, "location": "Paris, FR", "merchant_type": "alimentation"},
         "profile": {"avg_amount": 45, "home_city": "Paris, FR", "risk_history": 0.05}},
        {"label": "Achat crypto 3h du matin a Moscou", "tx": {"amount": 2400.0, "hour": 3, "location": "Moscou, RU", "merchant_type": "crypto", "is_new_device": True, "tx_velocity_1h": 4},
         "profile": {"avg_amount": 45, "home_city": "Paris, FR", "risk_history": 0.2}},
        {"label": "Cartes cadeaux 10x le montant habituel", "tx": {"amount": 980.0, "hour": 2, "location": "Paris, FR", "merchant_type": "cartes_cadeaux", "is_new_device": True},
         "profile": {"avg_amount": 45, "home_city": "Paris, FR", "risk_history": 0.1}},
    ]
    print("\nExemples de scoring :")
    for ex in examples:
        res = engine.predict(ex["tx"], ex["profile"])
        print(f"  {res['risk_score']:6.1f}/100  [{res['risk_level']:8}]  {ex['label']}")
        for factor in res["factors"][:2]:
            print(f"           - {factor}")


if __name__ == "__main__":
    args = sys.argv[1:]
    force_train = "--train" in args
    run_selftest = "--selftest" in args or not force_train

    samples = 12000
    if "--samples" in args:
        try:
            samples = int(args[args.index("--samples") + 1])
        except (IndexError, ValueError):
            pass

    if force_train:
        _engine = FraudEngine(auto_train=False)
        _engine.train(n_samples=samples)
    else:
        _engine = FraudEngine(auto_train=True)

    if run_selftest:
        _selftest(_engine)
