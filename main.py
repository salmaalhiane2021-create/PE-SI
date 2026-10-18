"""
main.py
=======
API REST du prototype FinTech "NovaPay AI".

Lancement :
    uvicorn main:app --reload --port 8000
    -> UI        : http://127.0.0.1:8000
    -> Swagger   : http://127.0.0.1:8000/docs

Points d'entree principaux :
    POST /api/v1/transaction                  -> scorer une transaction (IA)
    POST /api/v1/transaction/{id}/verify      -> valider l'OTP de l'authentification dynamique
    GET  /api/v1/dashboard/{user_id}          -> toutes les donnees du tableau de bord
    POST /api/v1/simulate                     -> generer un paiement normal ou suspect (demo)
"""

from __future__ import annotations

import json
import os
import random
import re
import secrets
import sqlite3
import string
import threading
from contextlib import closing
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ml_model import (
    FEATURE_LABELS,
    LOCATIONS,
    MERCHANT_CATEGORIES,
    OTP_THRESHOLD,
    FraudEngine,
    merchant_label,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("NOVAPAY_DB", os.path.join(BASE_DIR, "novapay_demo.db"))
INDEX_PATH = os.path.join(BASE_DIR, "index.html")

#: Seuil de bascule vers l'authentification dynamique (exigence : 30 %)
RISK_CHALLENGE_THRESHOLD = float(os.getenv("OTP_THRESHOLD", str(OTP_THRESHOLD)))
#: Blocage automatique (desactive par defaut : 101 = jamais)
AUTO_BLOCK_THRESHOLD = float(os.getenv("AUTO_BLOCK_RISK_THRESHOLD", "101"))
#: Duree de vie d'un OTP (secondes)
OTP_TTL_SECONDS = int(os.getenv("OTP_TTL_SECONDS", "180"))
OTP_MAX_ATTEMPTS = 3

#: Moteur IA (entraine au premier demarrage, puis recharge depuis ./artifacts)
engine = FraudEngine()
_db_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Multi-devises
# ---------------------------------------------------------------------------
#: Tous les montants sont stockes en EUR (devise de reference) puis convertis
#: a la volee selon la devise choisie dans l'interface.
#: `rate` = nombre d'unites de la devise pour 1 EUR (taux fixes pour la demo).
CURRENCIES: List[Dict[str, Any]] = [
    {"code": "EUR", "symbol": "\u20ac", "label": "Euro", "flag": "\U0001F1EA\U0001F1FA", "rate": 1.0},
    {"code": "USD", "symbol": "$", "label": "Dollar americain", "flag": "\U0001F1FA\U0001F1F8", "rate": 1.09},
    {"code": "MAD", "symbol": "MAD", "label": "Dirham marocain", "flag": "\U0001F1F2\U0001F1E6", "rate": 10.85},
    {"code": "GBP", "symbol": "\u00a3", "label": "Livre sterling", "flag": "\U0001F1EC\U0001F1E7", "rate": 0.84},
]
BASE_CURRENCY = "EUR"


def currency_info(code: Optional[str]) -> Dict[str, Any]:
    """Metadonnees d'une devise (retombe sur l'euro si inconnue)."""
    code = (code or BASE_CURRENCY).upper()
    for currency in CURRENCIES:
        if currency["code"] == code:
            return currency
    return CURRENCIES[0]


def to_display(amount_eur: float, code: str) -> float:
    """Convertit un montant EUR (stockage) vers la devise d'affichage."""
    return round(float(amount_eur or 0.0) * currency_info(code)["rate"], 2)


def to_base(amount: float, code: str) -> float:
    """Convertit un montant saisi dans la devise d'affichage vers l'euro (scoring + stockage)."""
    return round(float(amount or 0.0) / currency_info(code)["rate"], 2)


def convert_tx(tx: Dict[str, Any], code: str) -> Dict[str, Any]:
    """Ajoute a une transaction les montants convertis dans la devise demandee.

    Les montants cites dans les explications de l'IA ("... 2142.86 EUR")
    sont eux aussi reecrits dans la devise d'affichage.
    """
    info = currency_info(code)
    amount_eur = round(float(tx.get("amount") or 0.0), 2)
    tx["amount_eur"] = amount_eur
    tx["amount"] = round(amount_eur * info["rate"], 2)
    tx["currency"] = info["code"]
    tx["currency_symbol"] = info["symbol"]
    if info["code"] != BASE_CURRENCY and tx.get("factors"):
        tx["factors"] = [localize_amounts(factor, info) for factor in tx["factors"]]
    return tx


#: Detecte les montants en euros cites dans les explications : "(2142.86 EUR)"
_AMOUNT_IN_FACTOR = re.compile(r"\((\d+(?:[.,]\d+)?) EUR\)")


def localize_amounts(text: str, info: Dict[str, Any]) -> str:
    """Reecrit un montant EUR cite dans un texte vers la devise d'affichage."""
    def replace(match: "re.Match[str]") -> str:
        value = float(match.group(1).replace(",", ".")) * info["rate"]
        if value >= 1000:
            formatted = f"{value:,.0f}".replace(",", " ")
        else:
            formatted = f"{value:.2f}".replace(".", ",")
        return f"({formatted} {info['code']})"

    return _AMOUNT_IN_FACTOR.sub(replace, text)


def new_auth_code() -> str:
    """Code de securite a usage unique genere par l'IA (format OTP-1234)."""
    return f"OTP-{secrets.randbelow(9000) + 1000}"


def normalize_otp(value: Optional[str]) -> str:
    """Ne conserve que les chiffres du code (accepte 'OTP-8492', '8492', 'otp 8492'...)."""
    return re.sub(r"\D", "", str(value or ""))


#: Noms des pays (affichage) + villes disponibles par pays
COUNTRY_NAMES: Dict[str, str] = {
    "FR": "France", "BE": "Belgique", "CH": "Suisse", "GB": "Royaume-Uni", "US": "Etats-Unis",
    "CI": "Cote d'Ivoire", "SN": "Senegal", "MA": "Maroc", "NG": "Nigeria",
    "AE": "Emirats arabes unis", "HK": "Hong Kong", "RU": "Russie", "PA": "Panama",
}
COUNTRY_CITIES: Dict[str, List[str]] = {}
for _city, _meta in LOCATIONS.items():
    COUNTRY_CITIES.setdefault(_meta["country"], []).append(_city)
for _cities in COUNTRY_CITIES.values():
    _cities.sort()


def resolve_location(location: Optional[str], country: Optional[str], default: str) -> str:
    """Determine la ville de la transaction (parametre explicite > pays > ville de residence).

    Choix deterministe pour que l'apercu temps reel du risque reste stable.
    """
    if location:
        return location
    if country:
        cities = COUNTRY_CITIES.get(str(country).upper())
        if cities:
            return cities[0]
    return default

# ---------------------------------------------------------------------------
# Donnees de demonstration
# ---------------------------------------------------------------------------

DEMO_USERS: List[Dict[str, Any]] = [
    {
        "user_id": "user_001", "name": "Awa Diallo", "email": "awa.diallo@novapay.io",
        "phone": "+33 6 12 •• •• 42", "home_city": "Paris, FR", "avg_amount": 46.0,
        "monthly_income": 3250.0, "balance": 4185.75, "currency": "EUR",
        "card_last4": "4821", "accent": "#8b5cf6",
    },
    {
        "user_id": "user_002", "name": "Karim Benali", "email": "karim.benali@novapay.io",
        "phone": "+33 6 45 •• •• 07", "home_city": "Lyon, FR", "avg_amount": 128.0,
        "monthly_income": 4100.0, "balance": 9310.20, "currency": "EUR",
        "card_last4": "7734", "accent": "#22d3ee",
    },
    {
        "user_id": "user_003", "name": "Chloe Martin", "email": "chloe.martin@novapay.io",
        "phone": "+33 7 88 •• •• 19", "home_city": "Marseille, FR", "avg_amount": 27.0,
        "monthly_income": 2380.0, "balance": 1240.10, "currency": "EUR",
        "card_last4": "1096", "accent": "#f472b6",
    },
    {
        "user_id": "user_004", "name": "Moussa Traore", "email": "moussa.traore@novapay.io",
        "phone": "+33 6 70 •• •• 55", "home_city": "Lille, FR", "avg_amount": 74.0,
        "monthly_income": 3600.0, "balance": 5670.40, "currency": "EUR",
        "card_last4": "3319", "accent": "#34d399",
    },
]

MERCHANTS_BY_CATEGORY: Dict[str, List[str]] = {
    "alimentation": ["Carrefour City", "Monoprix", "Franprix", "Leclerc Express"],
    "restaurant": ["Le Petit Bistrot", "Starbucks", "Boulangerie Paul", "Sushi Shop"],
    "transport": ["SNCF Connect", "RATP", "Uber", "TotalEnergies Station"],
    "sante": ["Pharmacie Centrale", "Doctolib", "Laboratoire Bio"],
    "abonnements": ["Netflix", "Spotify", "Orange Telecom", "Free Mobile"],
    "mode": ["Zara", "Decathlon", "H&M", "Uniqlo"],
    "ecommerce": ["Amazon", "Cdiscount", "AliExpress", "Fnac.com"],
    "voyage": ["Booking.com", "Air France", "Airbnb", "Flixbus"],
    "electronique": ["Apple Store", "Boulanger", "Samsung Shop", "LDLC"],
    "jeux_argent": ["Winamax", "PMU", "Betclic"],
    "crypto": ["Binance", "Coinbase", "Kraken"],
    "cartes_cadeaux": ["Recharge PCS", "Transcash", "Paysafecard"],
    "transfert": ["Transfert P2P - Wero", "Lydia", "Western Union"],
}

# Categories plausibles pour une depense "normale" (ponderees)
NORMAL_CATEGORY_POOL = (
    ["alimentation"] * 8 + ["restaurant"] * 7 + ["transport"] * 5 + ["sante"] * 2
    + ["abonnements"] * 3 + ["mode"] * 3 + ["ecommerce"] * 3 + ["voyage"] * 1
)
# Categories typiques des fraudes
SUSPICIOUS_CATEGORY_POOL = ["crypto", "cartes_cadeaux", "jeux_argent", "electronique", "crypto", "cartes_cadeaux"]
# Villes a risque pour les simulations suspectes
SUSPICIOUS_CITIES = ["Moscou, RU", "Panama, PA", "Hong Kong, HK", "Lagos, NG", "Dubai, AE"]
LOW_RISK_CITIES = ["Paris, FR", "Lyon, FR", "Marseille, FR", "Lille, FR", "Bruxelles, BE"]

# ---------------------------------------------------------------------------
# Base de donnees SQLite
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id        TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    email          TEXT,
    phone          TEXT,
    home_city      TEXT,
    avg_amount     REAL,
    monthly_income REAL,
    balance        REAL,
    currency       TEXT,
    card_last4     TEXT,
    accent         TEXT,
    risk_history   REAL DEFAULT 0.1,
    created_at     TEXT
);

CREATE TABLE IF NOT EXISTS transactions (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    amount       REAL NOT NULL,
    currency     TEXT DEFAULT 'EUR',
    merchant     TEXT,
    merchant_type TEXT,
    location     TEXT,
    hour         INTEGER,
    channel      TEXT DEFAULT 'mobile_app',
    device_id    TEXT,
    is_new_device INTEGER DEFAULT 0,
    risk_score   REAL,
    risk_level   TEXT,
    decision     TEXT,
    status       TEXT,
    factors      TEXT,
    components   TEXT,
    auth_code    TEXT,
    is_anomaly   INTEGER DEFAULT 0,
    created_at   TEXT
);

CREATE TABLE IF NOT EXISTS otp_challenges (
    tx_id      TEXT PRIMARY KEY,
    otp_code   TEXT NOT NULL,
    created_at REAL,
    expires_at REAL,
    attempts   INTEGER DEFAULT 0,
    status     TEXT DEFAULT 'pending'
);

CREATE INDEX IF NOT EXISTS idx_tx_user_date ON transactions(user_id, created_at DESC);
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _db_lock, closing(get_conn()) as conn:
        conn.executescript(SCHEMA)
        # Migration douce : ajoute les colonnes manquantes sur une base existante
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(transactions)").fetchall()}
        for column, ddl in (("auth_code", "TEXT"),):
            if column not in columns:
                conn.execute(f"ALTER TABLE transactions ADD COLUMN {column} {ddl}")
        conn.commit()


# ---------------------------------------------------------------------------
# Helpers metier
# ---------------------------------------------------------------------------

def new_id(prefix: str, length: int = 10) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return f"{prefix}_{''.join(secrets.choice(alphabet) for _ in range(length))}"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def fetch_user(user_id: str) -> Dict[str, Any]:
    with closing(get_conn()) as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Utilisateur inconnu : {user_id}")
    return dict(row)


def fetch_all_users() -> List[Dict[str, Any]]:
    with closing(get_conn()) as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY user_id").fetchall()
    return [dict(r) for r in rows]


def tx_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    tx = dict(row)
    for key in ("factors", "components"):
        try:
            tx[key] = json.loads(tx.get(key) or "[]" if key == "factors" else tx.get(key) or "{}")
        except (TypeError, json.JSONDecodeError):
            tx[key] = [] if key == "factors" else {}
    tx["is_anomaly"] = bool(tx.get("is_anomaly"))
    tx["merchant_label"] = merchant_label(tx.get("merchant_type") or "")
    tx["merchant_icon"] = MERCHANT_CATEGORIES.get(tx.get("merchant_type") or "", {"icon": "box"})["icon"]
    return tx


def user_risk_history(user_id: str, limit: int = 12) -> float:
    """Historique de risque recent du client (0..1), pondere vers les plus recentes."""
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT risk_score FROM transactions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    if not rows:
        return 0.05
    weights, weighted = 0.0, 0.0
    for i, row in enumerate(rows):
        w = 0.85 ** i
        weighted += (float(row["risk_score"] or 0.0) / 100.0) * w
        weights += w
    return round(weighted / max(weights, 1e-6), 3)


def tx_velocity_1h(user_id: str) -> int:
    cutoff = (datetime.now() - timedelta(hours=1)).isoformat(timespec="seconds")
    with closing(get_conn()) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM transactions WHERE user_id = ? AND created_at >= ?",
            (user_id, cutoff),
        ).fetchone()
    return int(row["n"] or 0)


def security_score_for(user_id: str, limit: int = 25) -> Dict[str, Any]:
    """Indicateur dynamique 0-100 : plus le risque recent est fort, plus il baisse."""
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT risk_score, status, created_at FROM transactions "
            "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    if not rows:
        return {"score": 92.0, "level": "excellent", "incidents": 0, "avg_risk": 0.0}

    weighted, weights = 0.0, 0.0
    incidents = 0
    for i, row in enumerate(rows):
        w = 0.92 ** i
        weighted += float(row["risk_score"] or 0.0) * w
        weights += w
        if float(row["risk_score"] or 0.0) >= 70:
            incidents += 1

    avg_risk = weighted / max(weights, 1e-6)
    score = 100.0 - avg_risk * 0.72 - incidents * 3.2
    score = round(max(35.0, min(99.4, score)), 1)
    if score >= 85:
        level = "excellent"
    elif score >= 70:
        level = "bon"
    elif score >= 55:
        level = "moyen"
    else:
        level = "fragile"
    return {"score": score, "level": level, "incidents": incidents, "avg_risk": round(avg_risk, 1)}


def _insert_tx(conn: sqlite3.Connection, tx: Dict[str, Any]) -> None:
    tx = {**tx, "auth_code": tx.get("auth_code")}
    conn.execute(
        """INSERT INTO transactions (id, user_id, amount, currency, merchant, merchant_type, location,
               hour, channel, device_id, is_new_device, risk_score, risk_level, decision, status,
               factors, components, auth_code, is_anomaly, created_at)
           VALUES (:id, :user_id, :amount, :currency, :merchant, :merchant_type, :location, :hour,
               :channel, :device_id, :is_new_device, :risk_score, :risk_level, :decision, :status,
               :factors, :components, :auth_code, :is_anomaly, :created_at)""",
        tx,
    )


def update_balance(user_id: str, delta: float) -> float:
    with _db_lock, closing(get_conn()) as conn:
        conn.execute("UPDATE users SET balance = ROUND(balance + ?, 2) WHERE user_id = ?", (delta, user_id))
        row = conn.execute("SELECT balance FROM users WHERE user_id = ?", (user_id,)).fetchone()
        conn.commit()
    return float(row["balance"])


# ---------------------------------------------------------------------------
# Pipeline de decision (utilise par /transaction et /simulate)
# ---------------------------------------------------------------------------

def process_transaction(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Coeur du systeme : scoring IA -> decision -> MFA/OTP -> code d'autorisation.

    Le montant peut etre fourni dans la devise d'affichage (`currency`) :
    il est converti en EUR pour le scoring et le stockage, puis reconverti
    dans la reponse.
    """
    user = fetch_user(payload["user_id"])
    display_currency = currency_info(payload.get("currency"))["code"]
    amount_input = float(payload["amount"])
    if amount_input <= 0:
        raise HTTPException(status_code=422, detail="Le montant doit etre strictement positif.")
    amount = to_base(amount_input, display_currency)  # devise de reference = EUR

    merchant_type = payload.get("merchant_type") or "ecommerce"
    location = resolve_location(payload.get("location"), payload.get("country"), user["home_city"])
    hour = payload.get("hour")
    hour = int(hour) if hour is not None else datetime.now().hour
    device_id = payload.get("device_id") or f"dev_{user['user_id']}_principal"
    is_new_device = bool(payload.get("is_new_device")) or device_id != f"dev_{user['user_id']}_principal"
    channel = payload.get("channel") or "mobile_app"

    profile = {
        "home_city": user["home_city"],
        "avg_amount": user["avg_amount"],
        "risk_history": user_risk_history(user["user_id"]),
    }
    tx_input = {
        "amount": amount,
        "hour": hour,
        "location": location,
        "merchant_type": merchant_type,
        "is_new_device": is_new_device,
        "tx_velocity_1h": tx_velocity_1h(user["user_id"]),
        "is_weekend": int(datetime.now().weekday() >= 5),
    }

    # ---> Inference IA
    verdict = engine.predict(tx_input, profile)
    risk = float(verdict["risk_score"])
    tx_id = new_id("tx", 12)
    record = {
        "id": tx_id,
        "user_id": user["user_id"],
        "amount": round(amount, 2),
        "currency": BASE_CURRENCY,
        "merchant": payload.get("merchant") or random.choice(MERCHANTS_BY_CATEGORY.get(merchant_type, ["Marchand"])),
        "merchant_type": merchant_type,
        "location": location,
        "hour": hour,
        "channel": channel,
        "device_id": device_id,
        "is_new_device": int(is_new_device),
        "risk_score": risk,
        "risk_level": verdict["risk_level"],
        "decision": verdict["decision"],
        "status": "approved",
        "factors": json.dumps(verdict["factors"], ensure_ascii=False),
        "components": json.dumps(verdict["components"], ensure_ascii=False),
        "auth_code": None,
        "is_anomaly": int(risk >= 60),
        "created_at": now_iso(),
    }

    balance_after_eur: Optional[float] = None
    balance_after: Optional[float] = None
    otp_challenge: Optional[Dict[str, Any]] = None
    authorization_code: Optional[str] = None

    if risk >= AUTO_BLOCK_THRESHOLD:
        record["status"] = "declined"
        record["decision"] = "BLOCKED"
        message = "Transaction bloquee automatiquement par l'IA (risque critique)."
    elif risk >= RISK_CHALLENGE_THRESHOLD:
        # Authentification dynamique (MFA) : code a usage unique genere par l'IA
        record["status"] = "pending_auth"
        otp_code = new_auth_code()
        now_ts = datetime.now().timestamp()
        with _db_lock, closing(get_conn()) as conn:
            _insert_tx(conn, record)
            conn.execute(
                "INSERT OR REPLACE INTO otp_challenges (tx_id, otp_code, created_at, expires_at, attempts, status) "
                "VALUES (?, ?, ?, ?, 0, 'pending')",
                (tx_id, otp_code, now_ts, now_ts + OTP_TTL_SECONDS),
            )
            conn.commit()
        otp_challenge = {
            "tx_id": tx_id,
            "expires_in_seconds": OTP_TTL_SECONDS,
            "max_attempts": OTP_MAX_ATTEMPTS,
            "masked_phone": user["phone"],
            "channel": "SMS + Notification push",
            # Simulation uniquement : en production, l'OTP part par SMS et n'est jamais renvoye
            "demo_otp": otp_code,
            "code_format": "OTP-XXXX",
            "demo_note": "PoC : pas de passerelle SMS reelle, le code est affiche ici.",
        }
        message = (
            f"Risque {risk:.0f}% >= {RISK_CHALLENGE_THRESHOLD:.0f}% : "
            "authentification dynamique requise. Un code MFA a ete envoye."
        )
    else:
        # Risque faible : validation automatique + code d'autorisation a usage unique
        authorization_code = new_auth_code()
        record["auth_code"] = authorization_code
        with _db_lock, closing(get_conn()) as conn:
            _insert_tx(conn, record)
            conn.commit()
        balance_after_eur = update_balance(user["user_id"], -record["amount"])
        balance_after = to_display(balance_after_eur, display_currency)
        message = f"Risque {risk:.0f}% < {RISK_CHALLENGE_THRESHOLD:.0f}% : paiement valide automatiquement."

    # Explications IA : les montants cites sont reecrits dans la devise d'affichage
    factors = verdict["factors"]
    if display_currency != BASE_CURRENCY:
        factors = [localize_amounts(factor, currency_info(display_currency)) for factor in factors]

    return {
        "transaction_id": tx_id,
        "status": record["status"],
        "decision": record["decision"],
        "risk_score": risk,
        "risk_level": verdict["risk_level"],
        "risk_label": verdict["risk_label"],
        "threshold": RISK_CHALLENGE_THRESHOLD,
        "factors": factors,
        "components": verdict["components"],
        "top_features": verdict["top_features"],
        "features": verdict["features"],
        "requires_otp": record["status"] == "pending_auth",
        "otp_challenge": otp_challenge,
        "authorization_code": authorization_code,
        "message": message,
        "balance_after": balance_after,
        "balance_after_eur": balance_after_eur,
        "currency": display_currency,
        "rate": currency_info(display_currency)["rate"],
        "amount_eur": amount,
        "transaction": {
            "id": tx_id,
            "amount": to_display(amount, display_currency),
            "amount_eur": amount,
            "currency": display_currency,
            "currency_symbol": currency_info(display_currency)["symbol"],
            "merchant": record["merchant"],
            "merchant_type": merchant_type,
            "merchant_label": merchant_label(merchant_type),
            "location": location,
            "country": (LOCATIONS.get(location) or {}).get("country"),
            "hour": hour,
            "created_at": record["created_at"],
        },
    }


# ---------------------------------------------------------------------------
# Modeles Pydantic
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    user_id: str = Field(..., description="Identifiant utilisateur, ex: user_001")
    amount: float = Field(..., gt=0, description="Montant de la transaction (dans la devise `currency`)")
    merchant: Optional[str] = Field(None, description="Nom du marchand")
    merchant_type: str = Field("ecommerce", description="Categorie de marchand (cle de MERCHANT_CATEGORIES)")
    location: Optional[str] = Field(None, description="Cle de ville, ex: 'Paris, FR' (defaut : ville de residence)")
    country: Optional[str] = Field(None, description="Code pays a 2 lettres, ex: 'MA' (utilise si 'location' est absent)")
    currency: str = Field("EUR", description="Devise de saisie : EUR | USD | MAD | GBP")
    hour: Optional[int] = Field(None, ge=0, le=23, description="Heure de la transaction (0-23)")
    device_id: Optional[str] = Field(None, description="Identifiant appareil (nouveau = risque)")
    is_new_device: Optional[bool] = Field(None, description="Forcer 'appareil inconnu'")
    channel: str = Field("mobile_app", description="mobile_app | web | atm | api")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "user_001",
                    "amount": 12.5,
                    "merchant": "Boulangerie Paul",
                    "merchant_type": "restaurant",
                    "location": "Paris, FR",
                    "hour": 13,
                    "currency": "EUR",
                },
                {
                    "user_id": "user_001",
                    "amount": 2450.0,
                    "merchant": "Binance",
                    "merchant_type": "crypto",
                    "location": "Moscou, RU",
                    "hour": 3,
                    "is_new_device": True,
                    "currency": "USD",
                },
            ]
        }
    }


class PreviewRequest(BaseModel):
    """Apercu temps reel du risque (aucune transaction enregistree)."""
    user_id: str = Field(..., description="Identifiant utilisateur")
    amount: float = Field(..., gt=0, description="Montant dans la devise `currency`")
    merchant_type: str = Field("ecommerce", description="Categorie de marchand")
    location: Optional[str] = Field(None, description="Ville (cle de LOCATIONS)")
    country: Optional[str] = Field(None, description="Code pays si la ville n'est pas precisee")
    currency: str = Field("EUR", description="Devise de saisie")
    hour: Optional[int] = Field(None, ge=0, le=23)
    is_new_device: Optional[bool] = Field(None)


class OtpVerifyRequest(BaseModel):
    otp: str = Field(..., min_length=4, max_length=12, description="Code MFA a usage unique (ex: OTP-8492)")
    currency: str = Field("EUR", description="Devise d'affichage de la reponse")
    channel: Optional[str] = Field(None, description="Canal de validation (optionnel)")


class SimulateRequest(BaseModel):
    user_id: str = Field(..., description="Identifiant utilisateur")
    scenario: str = Field("random", description="normal | suspicious | random")
    currency: str = Field("EUR", description="Devise d'affichage : EUR | USD | MAD | GBP")


class AmountRequest(BaseModel):
    amount: float = Field(..., gt=0)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NovaPay AI - Paiement securise & detection de fraude",
    description=(
        "Prototype fintech : scoring de risque par IA (RandomForest + IsolationForest + regles) "
        "et authentification dynamique par OTP au-dela de 30 % de risque."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    seed_database()
    print("[API] NovaPay AI prete -> http://127.0.0.1:8000")


# ------------------------------------------------------------------ UI

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_PATH)


@app.get("/api/v1/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "novapay-ai",
        "version": "1.0.0",
        "model_loaded": engine.rf is not None,
        "challenge_threshold": RISK_CHALLENGE_THRESHOLD,
        "time": now_iso(),
    }


@app.get("/health", include_in_schema=False)
def health_alias() -> Dict[str, Any]:
    """Alias de /api/v1/health (utile pour les sondes de monitoring)."""
    return health()


# ------------------------------------------------------------ referentiels

@app.get("/api/v1/users")
def list_users(currency: str = Query("EUR", description="Devise d'affichage du solde")) -> List[Dict[str, Any]]:
    display = currency_info(currency)["code"]
    users = fetch_all_users()
    for user in users:
        user["balance_eur"] = round(float(user["balance"]), 2)
        user["balance"] = to_display(user["balance"], display)
        user["currency"] = display
        user["security"] = security_score_for(user["user_id"])
    return users


@app.get("/api/v1/currencies")
def list_currencies() -> Dict[str, Any]:
    """Devises supportees par l'interface (taux fixes pour la demo)."""
    return {"base": BASE_CURRENCY, "currencies": CURRENCIES}


@app.get("/api/v1/merchants")
def merchants() -> Dict[str, Any]:
    countries: Dict[str, Dict[str, Any]] = {}
    for city, meta in LOCATIONS.items():
        entry = countries.setdefault(meta["country"], {
            "code": meta["country"],
            "name": COUNTRY_NAMES.get(meta["country"], meta["country"]),
            "risk_hint": meta["risk_hint"],
            "cities": [],
        })
        entry["cities"].append(city)
        entry["risk_hint"] = round(max(entry["risk_hint"], meta["risk_hint"]), 2)
    return {
        "categories": [
            {"key": k, "label": v["label"], "risk": v["risk"], "icon": v["icon"],
             "merchants": MERCHANTS_BY_CATEGORY.get(k, [])}
            for k, v in sorted(MERCHANT_CATEGORIES.items(), key=lambda kv: kv[1]["risk"])
        ],
        "locations": [
            {"key": k, "risk_hint": v["risk_hint"], "country": v["country"],
             "country_name": COUNTRY_NAMES.get(v["country"], v["country"])}
            for k, v in sorted(LOCATIONS.items(), key=lambda kv: kv[1]["risk_hint"])
        ],
        "countries": sorted(countries.values(), key=lambda c: c["risk_hint"]),
    }


@app.get("/api/v1/model/insights")
def model_insights() -> Dict[str, Any]:
    info = engine.info()
    info["feature_labels"] = FEATURE_LABELS
    return info


# -------------------------------------------------------------- transactions

@app.post("/api/v1/transaction")
def create_transaction(request: TransactionRequest) -> Dict[str, Any]:
    """Soumet une transaction a l'IA.

    Reponse : score de risque, decision, et MFA (code a usage unique) si risque >= 30 %.
    Si la transaction est validee (automatiquement ou apres MFA), un code
    d'autorisation a usage unique `OTP-XXXX` est genere par l'IA.
    """
    return process_transaction(request.model_dump())


@app.post("/api/v1/transaction/preview")
def preview_transaction(request: PreviewRequest) -> Dict[str, Any]:
    """Apercu temps reel du score de risque (aucune donnee enregistree).

    Utilise par la modale de simulation pour afficher le risque pendant la saisie.
    """
    user = fetch_user(request.user_id)
    location = resolve_location(request.location, request.country, user["home_city"])
    profile = {
        "home_city": user["home_city"],
        "avg_amount": user["avg_amount"],
        "risk_history": user_risk_history(user["user_id"]),
    }
    verdict = engine.predict(
        {
            "amount": to_base(request.amount, request.currency),
            "hour": request.hour,
            "location": location,
            "merchant_type": request.merchant_type,
            "is_new_device": bool(request.is_new_device),
            "tx_velocity_1h": tx_velocity_1h(user["user_id"]),
            "is_weekend": int(datetime.now().weekday() >= 5),
        },
        profile,
    )
    display = currency_info(request.currency)["code"]
    factors = verdict["factors"]
    if display != BASE_CURRENCY:
        factors = [localize_amounts(factor, currency_info(display)) for factor in factors]
    return {
        "risk_score": verdict["risk_score"],
        "risk_level": verdict["risk_level"],
        "risk_label": verdict["risk_label"],
        "decision": verdict["decision"],
        "requires_otp": verdict["requires_otp"],
        "threshold": RISK_CHALLENGE_THRESHOLD,
        "factors": factors,
        "components": verdict["components"],
        "location": location,
        "amount_eur": to_base(request.amount, request.currency),
        "currency": currency_info(request.currency)["code"],
    }


@app.post("/api/v1/transaction/{tx_id}/verify")
def verify_otp(tx_id: str, request: OtpVerifyRequest) -> Dict[str, Any]:
    """Valide le code MFA declenche par l'authentification dynamique."""
    code = normalize_otp(request.otp)
    with _db_lock, closing(get_conn()) as conn:
        tx = conn.execute("SELECT * FROM transactions WHERE id = ?", (tx_id,)).fetchone()
        if tx is None:
            raise HTTPException(status_code=404, detail="Transaction inconnue.")
        if tx["status"] != "pending_auth":
            raise HTTPException(status_code=409, detail=f"Transaction deja traitee (statut: {tx['status']}).")

        challenge = conn.execute("SELECT * FROM otp_challenges WHERE tx_id = ?", (tx_id,)).fetchone()
        if challenge is None:
            raise HTTPException(status_code=409, detail="Aucun challenge MFA pour cette transaction.")
        if challenge["status"] == "locked":
            raise HTTPException(status_code=423, detail="Trop de tentatives : veuillez relancer la transaction.")

        now_ts = datetime.now().timestamp()
        if now_ts > float(challenge["expires_at"]):
            conn.execute("UPDATE otp_challenges SET status='expired' WHERE tx_id = ?", (tx_id,))
            conn.execute("UPDATE transactions SET status='expired', decision='EXPIRED' WHERE id = ?", (tx_id,))
            conn.commit()
            raise HTTPException(status_code=410, detail="Code MFA expire. Relancez la transaction.")

        if code != normalize_otp(challenge["otp_code"]):
            attempts = int(challenge["attempts"]) + 1
            status = "locked" if attempts >= OTP_MAX_ATTEMPTS else "pending"
            conn.execute("UPDATE otp_challenges SET attempts = ?, status = ? WHERE tx_id = ?", (attempts, status, tx_id))
            if status == "locked":
                conn.execute("UPDATE transactions SET status='declined', decision='BLOCKED' WHERE id = ?", (tx_id,))
            conn.commit()
            remaining = max(0, OTP_MAX_ATTEMPTS - attempts)
            raise HTTPException(
                status_code=401,
                detail=f"Code MFA invalide. Tentatives restantes : {remaining}."
                + (" Transaction bloquee par securite." if status == "locked" else ""),
            )

        # Code correct -> transaction approuvee, le code devient le code d'autorisation
        auth_code = str(challenge["otp_code"])
        conn.execute("UPDATE otp_challenges SET status='used' WHERE tx_id = ?", (tx_id,))
        conn.execute(
            "UPDATE transactions SET status='approved', decision='ACCEPTED_AFTER_OTP', auth_code = ? WHERE id = ?",
            (auth_code, tx_id),
        )
        conn.commit()
        user_id, amount = tx["user_id"], float(tx["amount"])
        tx_payload = tx_row_to_dict(tx)

    display = currency_info(request.currency)["code"]
    balance_after_eur = update_balance(user_id, -amount)
    security = security_score_for(user_id)
    return {
        "transaction_id": tx_id,
        "status": "approved",
        "decision": "ACCEPTED_AFTER_OTP",
        "message": "Code MFA valide : authentification dynamique reussie, paiement autorise.",
        "authorization_code": auth_code,
        "currency": display,
        "amount": to_display(amount, display),
        "amount_eur": amount,
        "balance_after": to_display(balance_after_eur, display),
        "balance_after_eur": balance_after_eur,
        "security_score": security["score"],
        "transaction": convert_tx(tx_payload, display),
    }


@app.post("/api/v1/transaction/{tx_id}/resend")
def resend_otp(tx_id: str, currency: str = Query("EUR", description="Devise d'affichage")) -> Dict[str, Any]:
    """Genere un nouveau code MFA (simulation d'un renvoi SMS)."""
    with _db_lock, closing(get_conn()) as conn:
        tx = conn.execute("SELECT * FROM transactions WHERE id = ?", (tx_id,)).fetchone()
        if tx is None or tx["status"] != "pending_auth":
            raise HTTPException(status_code=404, detail="Aucune authentification en attente pour cette transaction.")
        otp_code = new_auth_code()
        now_ts = datetime.now().timestamp()
        conn.execute(
            "UPDATE otp_challenges SET otp_code = ?, created_at = ?, expires_at = ?, attempts = 0, status='pending' "
            "WHERE tx_id = ?",
            (otp_code, now_ts, now_ts + OTP_TTL_SECONDS, tx_id),
        )
        conn.commit()
        tx_payload = tx_row_to_dict(tx)
    display = currency_info(currency)["code"]
    return {
        "transaction_id": tx_id,
        "message": "Nouveau code de securite envoye (simulation SMS).",
        "expires_in_seconds": OTP_TTL_SECONDS,
        "demo_otp": otp_code,
        "code_format": "OTP-XXXX",
        "transaction": convert_tx(tx_payload, display),
    }


@app.post("/api/v1/transaction/{tx_id}/cancel")
def cancel_transaction(tx_id: str) -> Dict[str, Any]:
    """L'utilisateur refuse la transaction -> annulee."""
    with _db_lock, closing(get_conn()) as conn:
        tx = conn.execute("SELECT * FROM transactions WHERE id = ?", (tx_id,)).fetchone()
        if tx is None:
            raise HTTPException(status_code=404, detail="Transaction inconnue.")
        if tx["status"] != "pending_auth":
            raise HTTPException(status_code=409, detail=f"Transaction deja traitee (statut: {tx['status']}).")
        conn.execute("UPDATE transactions SET status='declined', decision='CANCELLED_BY_USER' WHERE id = ?", (tx_id,))
        conn.execute("UPDATE otp_challenges SET status='cancelled' WHERE tx_id = ?", (tx_id,))
        conn.commit()
    return {"transaction_id": tx_id, "status": "declined", "message": "Transaction annulee par l'utilisateur (aucun debit)."}


@app.get("/api/v1/transactions")
def list_transactions(
    user_id: str = Query(..., description="Identifiant utilisateur"),
    limit: int = Query(20, ge=1, le=200),
    only_flagged: bool = Query(False, description="Uniquement les transactions a risque"),
    currency: str = Query("EUR", description="Devise d'affichage : EUR | USD | MAD | GBP"),
) -> Dict[str, Any]:
    query = "SELECT * FROM transactions WHERE user_id = ?"
    params: List[Any] = [user_id]
    if only_flagged:
        query += " AND risk_score >= 30"
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with closing(get_conn()) as conn:
        rows = conn.execute(query, params).fetchall()
    display = currency_info(currency)["code"]
    return {
        "user_id": user_id,
        "count": len(rows),
        "currency": display,
        "transactions": [convert_tx(tx_row_to_dict(r), display) for r in rows],
    }


@app.get("/api/v1/alerts/{user_id}")
def alerts(
    user_id: str,
    limit: int = Query(3, ge=1, le=20, description="Nombre de tentatives de fraude a retourner"),
    currency: str = Query("EUR", description="Devise d'affichage"),
) -> Dict[str, Any]:
    """Historique des dernieres tentatives de fraude : heure, montant et motif du blocage."""
    fetch_user(user_id)  # 404 si utilisateur inconnu
    with closing(get_conn()) as conn:
        rows = conn.execute(
            "SELECT * FROM transactions WHERE user_id = ? AND (risk_score >= 60 OR status IN ('declined', 'expired')) "
            "ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    display = currency_info(currency)["code"]
    entries: List[Dict[str, Any]] = []
    for row in rows:
        tx = convert_tx(tx_row_to_dict(row), display)
        factors = tx.get("factors") or []
        status = row["status"]
        decision = row["decision"]
        if status == "declined":
            action = "Transaction bloquée" if decision == "BLOCKED" else "Refusée par le client"
        elif status == "expired":
            action = "Code MFA expiré — transaction abandonnée"
        else:
            action = "Validée après authentification MFA (code à usage unique)"
        entries.append({
            "id": tx["id"],
            "created_at": tx["created_at"],
            "hour": tx["hour"],
            "amount": tx["amount"],
            "amount_eur": tx["amount_eur"],
            "currency": display,
            "merchant": tx["merchant"],
            "merchant_label": tx["merchant_label"],
            "merchant_icon": tx["merchant_icon"],
            "location": tx["location"],
            "risk_score": tx["risk_score"],
            "risk_level": tx["risk_level"],
            "status": status,
            "decision": decision,
            "action": action,
            "reason": " · ".join(factors[:2]) if factors else "Comportement atypique détecté par l'IA",
            "factors": factors,
        })
    return {"user_id": user_id, "count": len(entries), "currency": display, "alerts": entries}


# ------------------------------------------------------------------ dashboard

@app.get("/api/v1/dashboard/{user_id}")
def dashboard(
    user_id: str,
    tx_limit: int = Query(14, ge=3, le=100),
    currency: str = Query("EUR", description="Devise d'affichage : EUR | USD | MAD | GBP"),
) -> Dict[str, Any]:
    """Toutes les donnees du tableau de bord : solde, KPI, graphiques, anomalies, IA.

    Tous les montants (stockes en EUR) sont convertis dans la devise demandee.
    """
    user = fetch_user(user_id)
    display = currency_info(currency)["code"]
    rate = currency_info(display)["rate"]
    since_30d = (datetime.now() - timedelta(days=30)).isoformat(timespec="seconds")

    with closing(get_conn()) as conn:
        recent = conn.execute(
            "SELECT * FROM transactions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, tx_limit),
        ).fetchall()
        month_rows = conn.execute(
            "SELECT * FROM transactions WHERE user_id = ? AND created_at >= ? AND status = 'approved'",
            (user_id, since_30d),
        ).fetchall()
        all_rows = conn.execute(
            "SELECT id, amount, merchant_type, risk_score, status, created_at, hour FROM transactions "
            "WHERE user_id = ? ORDER BY created_at DESC LIMIT 400",
            (user_id,),
        ).fetchall()

    # --- Depenses par categorie (30 jours)
    by_category: Dict[str, Dict[str, Any]] = {}
    for row in month_rows:
        key = row["merchant_type"] or "autre"
        bucket = by_category.setdefault(key, {"category": key, "label": merchant_label(key), "amount": 0.0, "count": 0})
        bucket["amount"] = round(bucket["amount"] + float(row["amount"]), 2)
        bucket["count"] += 1
    spending_by_category = sorted(by_category.values(), key=lambda b: b["amount"], reverse=True)
    for bucket in spending_by_category:
        bucket["amount"] = round(bucket["amount"] * rate, 2)

    # --- Activite par heure + tendance journaliere (14 jours)
    activity_by_hour = [0] * 24
    daily: Dict[str, Dict[str, Any]] = {}
    anomalies: List[Dict[str, Any]] = []
    for row in all_rows:
        hour = int(row["hour"] or 0)
        activity_by_hour[hour] += 1
        day = (row["created_at"] or "")[:10]
        if day >= (datetime.now() - timedelta(days=13)).date().isoformat():
            d = daily.setdefault(day, {"date": day, "amount": 0.0, "anomalies": 0, "risk_sum": 0.0, "count": 0})
            if row["status"] in ("approved", "pending_auth"):
                d["amount"] = round(d["amount"] + float(row["amount"]), 2)
            d["count"] += 1
            d["risk_sum"] += float(row["risk_score"] or 0.0)
            if float(row["risk_score"] or 0.0) >= 60:
                d["anomalies"] += 1
        if float(row["risk_score"] or 0.0) >= 60:
            anomalies.append({"id": row["id"], "risk_score": float(row["risk_score"]),
                              "amount": to_display(float(row["amount"]), display),
                              "amount_eur": float(row["amount"]),
                              "currency": display, "created_at": row["created_at"],
                              "merchant_type": row["merchant_type"]})

    daily_trend = []
    for offset in range(13, -1, -1):
        day = (datetime.now() - timedelta(days=offset)).date().isoformat()
        entry = daily.get(day)
        daily_trend.append({
            "date": day,
            "amount": round(entry["amount"] * rate, 2) if entry else 0.0,
            "avg_risk": round(entry["risk_sum"] / entry["count"], 1) if entry and entry["count"] else 0.0,
            "anomalies": entry["anomalies"] if entry else 0,
        })

    approved_month = [r for r in month_rows]
    spend_30d_eur = round(sum(float(r["amount"]) for r in approved_month), 2)
    risk_scores_30d = [float(r["risk_score"] or 0) for r in approved_month]
    challenges = [convert_tx(tx_row_to_dict(r), display) for r in recent if r["status"] == "pending_auth"]

    return {
        "display_currency": display,
        "currency_meta": currency_info(display),
        "rate": rate,
        "user": {
            "user_id": user["user_id"], "name": user["name"], "email": user["email"], "phone": user["phone"],
            "home_city": user["home_city"], "country": (LOCATIONS.get(user["home_city"]) or {}).get("country"),
            "card_last4": user["card_last4"], "accent": user["accent"],
            "avg_amount": to_display(user["avg_amount"], display),
            "avg_amount_eur": user["avg_amount"],
            "monthly_income": to_display(user["monthly_income"], display),
        },
        "balance": {
            "amount": to_display(user["balance"], display),
            "amount_eur": round(float(user["balance"]), 2),
            "currency": display,
            "symbol": currency_info(display)["symbol"],
        },
        "kpis": {
            "spend_30d": to_display(spend_30d_eur, display),
            "income_30d": to_display(user["monthly_income"], display),
            "tx_count_30d": len(approved_month),
            "avg_ticket": to_display(spend_30d_eur / len(approved_month), display) if approved_month else 0.0,
            "avg_risk_30d": round(sum(risk_scores_30d) / len(risk_scores_30d), 1) if risk_scores_30d else 0.0,
            "alerts_30d": sum(1 for s in risk_scores_30d if s >= 60),
            "pending_challenges": len(challenges),
        },
        "security": security_score_for(user_id),
        "spending_by_category": spending_by_category,
        "activity_by_hour": activity_by_hour,
        "daily_trend": daily_trend,
        "recent_transactions": [convert_tx(tx_row_to_dict(r), display) for r in recent],
        "anomalies": anomalies[:6],
        "pending_challenges": challenges,
        "model": {
            "name": "Hybride RandomForest + IsolationForest + Regles",
            "threshold": RISK_CHALLENGE_THRESHOLD,
            "auc": engine.metrics.get("auc_roc"),
            "feature_importances": engine.metrics.get("feature_importances", {}),
        },
    }


@app.get("/api/v1/analytics/{user_id}")
def analytics(user_id: str, currency: str = Query("EUR", description="Devise d'affichage")) -> Dict[str, Any]:
    """Vue analytique dediee (categories, rythme horaire, anomalies, distribution du risque)."""
    data = dashboard(user_id, currency=currency)
    buckets = {"faible": 0, "modere": 0, "eleve": 0, "critique": 0}
    for tx in data["recent_transactions"]:
        buckets[tx["risk_level"]] = buckets.get(tx["risk_level"], 0) + 1
    return {
        "currency": data["display_currency"],
        "spending_by_category": data["spending_by_category"],
        "activity_by_hour": data["activity_by_hour"],
        "daily_trend": data["daily_trend"],
        "anomalies": data["anomalies"],
        "risk_distribution": buckets,
        "security": data["security"],
        "model": data["model"],
    }


# ------------------------------------------------------------------ simulation

@app.post("/api/v1/simulate")
def simulate(request: SimulateRequest) -> Dict[str, Any]:
    """Genere un paiement coherent et le passe dans le pipeline IA (mode demo)."""
    user = fetch_user(request.user_id)
    display = currency_info(request.currency)["code"]
    rate = currency_info(display)["rate"]
    scenario = request.scenario if request.scenario in ("normal", "suspicious") else random.choice(["normal", "suspicious"])
    avg = max(float(user["avg_amount"]), 10.0)
    now_hour = datetime.now().hour

    if scenario == "normal":
        category = random.choice(NORMAL_CATEGORY_POOL)
        amount_eur = round(avg * random.uniform(0.35, 1.7), 2)
        location = user["home_city"] if random.random() < 0.9 else random.choice([c for c in LOW_RISK_CITIES if c != user["home_city"]])
        payload = {
            "user_id": user["user_id"], "amount": round(amount_eur * rate, 2), "currency": display,
            "merchant": random.choice(MERCHANTS_BY_CATEGORY.get(category, ["Marchand local"])),
            "merchant_type": category, "location": location, "hour": now_hour,
            "device_id": f"dev_{user['user_id']}_principal", "is_new_device": False, "channel": "mobile_app",
        }
    else:
        category = random.choice(SUSPICIOUS_CATEGORY_POOL)
        amount_eur = round(avg * random.uniform(6.0, 15.0), 2)
        payload = {
            "user_id": user["user_id"], "amount": round(amount_eur * rate, 2), "currency": display,
            "merchant": random.choice(MERCHANTS_BY_CATEGORY.get(category, ["Marchand"])),
            "merchant_type": category, "location": random.choice(SUSPICIOUS_CITIES),
            "hour": random.choice([1, 2, 3, 4, 23]),
            "device_id": f"dev_inconnu_{secrets.token_hex(4)}", "is_new_device": True, "channel": "web",
        }

    result = process_transaction(payload)
    return {"scenario": scenario, "currency": display, "payload": payload, "result": result}


# ------------------------------------------------------------------ demo reset

@app.post("/api/v1/users/{user_id}/reset")
def reset_user(user_id: str) -> Dict[str, Any]:
    """Reinitialise l'historique + le solde d'un utilisateur (utile en demo)."""
    user = fetch_user(user_id)
    default_balance = next(u["balance"] for u in DEMO_USERS if u["user_id"] == user_id)
    with _db_lock, closing(get_conn()) as conn:
        conn.execute("DELETE FROM transactions WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM otp_challenges WHERE tx_id NOT IN (SELECT id FROM transactions)")
        conn.execute("UPDATE users SET balance = ? WHERE user_id = ?", (default_balance, user_id))
        conn.commit()
    seed_history_for_user(user)
    return {"user_id": user_id, "message": "Historique reinitialise.", "security": security_score_for(user_id)}


# ---------------------------------------------------------------------------
# Generation des donnees de demonstration
# ---------------------------------------------------------------------------

def _random_history_tx(user: Dict[str, Any], moment: datetime, suspicious: bool = False) -> Dict[str, Any]:
    avg = max(float(user["avg_amount"]), 10.0)
    if suspicious:
        category = random.choice(SUSPICIOUS_CATEGORY_POOL)
        amount = round(avg * random.uniform(4.5, 11.0), 2)
        location = random.choice(SUSPICIOUS_CITIES)
        hour = random.choice([1, 2, 3, 4, 23])
        device = f"dev_inconnu_{secrets.token_hex(3)}"
    else:
        category = random.choice(NORMAL_CATEGORY_POOL)
        amount = round(avg * random.lognormvariate(-0.1, 0.45), 2)
        location = user["home_city"] if random.random() < 0.88 else random.choice(LOW_RISK_CITIES)
        hour = int(max(6, min(22, random.gauss(14, 4))))
        device = f"dev_{user['user_id']}_principal"
    merchant = random.choice(MERCHANTS_BY_CATEGORY.get(category, ["Marchand local"]))
    return {
        "user_id": user["user_id"],
        "amount": amount,
        "currency": user["currency"],
        "merchant": merchant,
        "merchant_type": category,
        "location": location,
        "hour": hour,
        "channel": random.choice(["mobile_app", "mobile_app", "web", "atm"]),
        "device_id": device,
        "is_new_device": int(device != f"dev_{user['user_id']}_principal"),
        "created_at": moment.isoformat(timespec="seconds"),
    }


def seed_history_for_user(user: Dict[str, Any]) -> int:
    """Cree ~45 jours d'historique realiste (dont quelques anomalies volontaires)."""
    profile = {"home_city": user["home_city"], "avg_amount": user["avg_amount"], "risk_history": 0.08}
    rows: List[Dict[str, Any]] = []
    today = datetime.now()

    for day_offset in range(45, -1, -1):
        day = today - timedelta(days=day_offset)
        for _ in range(random.choice([0, 1, 1, 2, 2, 3])):
            moment = day.replace(hour=random.randint(7, 22), minute=random.randint(0, 59), second=0, microsecond=0)
            if moment > today:
                continue
            base = _random_history_tx(user, moment)
            rows.append(base)

    # Anomalies volontaires (visibles sur le graphique)
    for _ in range(random.randint(2, 4)):
        day = today - timedelta(days=random.randint(3, 40))
        moment = day.replace(hour=random.choice([2, 3, 4]), minute=random.randint(0, 59), second=0, microsecond=0)
        rows.append(_random_history_tx(user, moment, suspicious=True))

    rows.sort(key=lambda r: r["created_at"])
    inserted = 0
    with _db_lock, closing(get_conn()) as conn:
        for row in rows:
            verdict = engine.predict(
                {
                    "amount": row["amount"], "hour": row["hour"], "location": row["location"],
                    "merchant_type": row["merchant_type"], "is_new_device": bool(row["is_new_device"]),
                    "tx_velocity_1h": 0, "is_weekend": 0,
                },
                profile,
            )
            risk = float(verdict["risk_score"])
            # Les transactions passees sont supposees avoir ete traitees (validees ou refusees)
            if risk >= 85:
                status, decision, auth_code = "declined", "BLOCKED", None
            elif risk >= RISK_CHALLENGE_THRESHOLD:
                status, decision, auth_code = "approved", "ACCEPTED_AFTER_OTP", new_auth_code()
            else:
                status, decision, auth_code = "approved", "ACCEPTED", new_auth_code()
            _insert_tx(conn, {
                "id": new_id("tx", 12), "user_id": row["user_id"], "amount": row["amount"],
                "currency": row["currency"], "merchant": row["merchant"], "merchant_type": row["merchant_type"],
                "location": row["location"], "hour": row["hour"], "channel": row["channel"],
                "device_id": row["device_id"], "is_new_device": row["is_new_device"],
                "risk_score": risk, "risk_level": verdict["risk_level"], "decision": decision, "status": status,
                "factors": json.dumps(verdict["factors"], ensure_ascii=False),
                "components": json.dumps(verdict["components"], ensure_ascii=False),
                "auth_code": auth_code,
                "is_anomaly": int(risk >= 60), "created_at": row["created_at"],
            })
            inserted += 1
        conn.commit()
    return inserted


def seed_database() -> None:
    """Cree les utilisateurs de demo et leur historique au premier lancement."""
    init_db()
    with _db_lock, closing(get_conn()) as conn:
        existing = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
        if not existing:
            for user in DEMO_USERS:
                conn.execute(
                    """INSERT INTO users (user_id, name, email, phone, home_city, avg_amount, monthly_income,
                           balance, currency, card_last4, accent, risk_history, created_at)
                       VALUES (:user_id, :name, :email, :phone, :home_city, :avg_amount, :monthly_income,
                           :balance, :currency, :card_last4, :accent, 0.1, :created_at)""",
                    {**user, "created_at": now_iso()},
                )
            conn.commit()
    for user in fetch_all_users():
        with closing(get_conn()) as conn:
            count = conn.execute("SELECT COUNT(*) AS n FROM transactions WHERE user_id = ?", (user["user_id"],)).fetchone()["n"]
        if not count:
            n = seed_history_for_user(user)
            print(f"[API] Historique genere pour {user['name']} : {n} transactions")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
