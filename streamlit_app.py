"""
streamlit_app.py
================
NovaPay AI — version **Streamlit** du prototype FinTech (deployable sur
Streamlit Community Cloud : https://share.streamlit.io).

Lancement local :
    streamlit run streamlit_app.py

Le moteur IA est celui de ``ml_model.py`` (RandomForest + IsolationForest +
regles metier, artefacts pre-entraines charges depuis ``artifacts/``).
Les donnees de demonstration proviennent de ``demo_data.json``, genere par
``export_demo.py`` a partir de la base SQLite (chargement instantane, pas de
re-entrainement au demarrage).

Fonctionnalites :
    * tableau de bord (solde, KPI, graphiques, anomalies, score de securite)
    * multi-devises (EUR / USD / MAD / GBP)
    * simulateur de paiement avec score de risque **en temps reel**
    * authentification dynamique : code a usage unique ``OTP-XXXX``
    * historique des alertes (heur, montant, motif du blocage)
"""

from __future__ import annotations

import json
import os
import random
import re
import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml_model import (
    FEATURE_LABELS,
    LOCATIONS,
    MERCHANT_CATEGORIES,
    OTP_THRESHOLD,
    FraudEngine,
    merchant_label,
)

# ===========================================================================
# Configuration
# ===========================================================================

BASE_CURRENCY = "EUR"
RISK_CHALLENGE_THRESHOLD = float(OTP_THRESHOLD)                     # 30 % par defaut
AUTO_BLOCK_THRESHOLD = float(os.getenv("AUTO_BLOCK_RISK_THRESHOLD", "101"))
OTP_MAX_ATTEMPTS = 3
OTP_TTL_SECONDS = int(os.getenv("OTP_TTL_SECONDS", "180"))
DATA_FILE = Path(__file__).resolve().parent / "demo_data.json"

#: Devises d'affichage (taux fixes de demonstration, base EUR)
CURRENCIES: List[Dict[str, Any]] = [
    {"code": "EUR", "symbol": "€", "label": "Euro", "rate": 1.0},
    {"code": "USD", "symbol": "$", "label": "Dollar US", "rate": 1.09},
    {"code": "MAD", "symbol": "MAD", "label": "Dirham marocain", "rate": 10.85},
    {"code": "GBP", "symbol": "£", "label": "Livre sterling", "rate": 0.84},
]

COUNTRY_NAMES: Dict[str, str] = {
    "FR": "France", "BE": "Belgique", "CH": "Suisse", "GB": "Royaume-Uni", "US": "États-Unis",
    "CI": "Côte d'Ivoire", "SN": "Sénégal", "MA": "Maroc", "NG": "Nigeria",
    "AE": "Émirats arabes unis", "HK": "Hong Kong", "RU": "Russie", "PA": "Panama",
}

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

NORMAL_CATEGORY_POOL = (
    ["alimentation"] * 8 + ["restaurant"] * 7 + ["transport"] * 5 + ["sante"] * 2
    + ["abonnements"] * 3 + ["mode"] * 3 + ["ecommerce"] * 3 + ["voyage"] * 1
)
SUSPICIOUS_CATEGORY_POOL = ["crypto", "cartes_cadeaux", "jeux_argent", "electronique"]
SUSPICIOUS_CITIES = ["Moscou, RU", "Panama, PA", "Hong Kong, HK", "Lagos, NG", "Dubai, AE"]

CAT_CHART_COLORS = ["#8b5cf6", "#22d3ee", "#34d399", "#fbbf24", "#f472b6",
                    "#60a5fa", "#fb923c", "#a3e635", "#f87171", "#c084fc"]

#: Villes disponibles par pays (choix deterministe, comme dans main.py)
COUNTRY_CITIES: Dict[str, List[str]] = {}
for _city, _meta in LOCATIONS.items():
    COUNTRY_CITIES.setdefault(_meta["country"], []).append(_city)
for _cities in COUNTRY_CITIES.values():
    _cities.sort()

COUNTRIES = sorted(
    ({  "code": code,
        "name": COUNTRY_NAMES.get(code, code),
        "risk_hint": max(LOCATIONS[c]["risk_hint"] for c in cities),
        "cities": cities,
    } for code, cities in COUNTRY_CITIES.items()),
    key=lambda c: c["risk_hint"],
)

# ===========================================================================
# Helpers
# ===========================================================================


def currency_info(code: Optional[str]) -> Dict[str, Any]:
    code = (code or BASE_CURRENCY).upper()
    for currency in CURRENCIES:
        if currency["code"] == code:
            return currency
    return CURRENCIES[0]


def to_display(amount_eur: float, code: str) -> float:
    return round(float(amount_eur or 0.0) * currency_info(code)["rate"], 2)


def to_base(amount: float, code: str) -> float:
    return round(float(amount or 0.0) / currency_info(code)["rate"], 2)


def fmt_money(amount: float, code: str) -> str:
    info = currency_info(code)
    text = f"{float(amount or 0.0):,.2f}".replace(",", " ").replace(".", ",")
    if info["symbol"] in ("€", "$", "£"):
        return f"{text} {info['symbol']}"
    return f"{text} {info['code']}"


_AMOUNT_IN_FACTOR = re.compile(r"\((\d+(?:[.,]\d+)?) EUR\)")


def localize_amounts(text: str, code: str) -> str:
    """Reecrit les montants EUR cites dans une explication IA dans la devise d'affichage."""
    info = currency_info(code)
    if info["code"] == BASE_CURRENCY:
        return text

    def replace(match: "re.Match[str]") -> str:
        value = float(match.group(1).replace(",", ".")) * info["rate"]
        formatted = f"{value:,.0f}".replace(",", " ") if value >= 1000 else f"{value:.2f}".replace(".", ",")
        return f"({formatted} {info['code']})"

    return _AMOUNT_IN_FACTOR.sub(replace, text)


def localize_factors(factors: List[str], code: str) -> List[str]:
    return [localize_amounts(factor, code) for factor in factors]


def new_id(prefix: str, length: int = 10) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return f"{prefix}_{''.join(secrets.choice(alphabet) for _ in range(length))}"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_auth_code() -> str:
    """Code de securite a usage unique genere par l'IA (format OTP-1234)."""
    return f"OTP-{secrets.randbelow(9000) + 1000}"


def normalize_otp(value: Optional[str]) -> str:
    return re.sub(r"\D", "", str(value or ""))


def resolve_location(country: Optional[str], default: str) -> str:
    if country:
        cities = COUNTRY_CITIES.get(str(country).upper())
        if cities:
            return cities[0]
    return default


def risk_color(score: float) -> str:
    return "#34d399" if score < 30 else "#fbbf24" if score < 60 else "#fb923c" if score < 85 else "#f43f5e"


def risk_badge(score: float, label: str) -> str:
    color = risk_color(score)
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color}66;'
        f'border-radius:10px;padding:2px 10px;font-size:0.75rem;font-weight:700;">'
        f'{label} · {score:.0f} %</span>'
    )


def status_label(status: str) -> str:
    return {"approved": "Validée", "pending_auth": "En attente MFA",
            "declined": "Bloquée", "expired": "Expirée"}.get(status, status)


#: Libelles accentues (ml_model reste volontairement en ASCII pour la console)
RISK_LEVEL_LABELS = {
    "faible": "Risque faible",
    "modere": "Risque modéré",
    "eleve": "Risque élevé",
    "critique": "Risque critique",
}


def risk_level_label(level: str, fallback: str = "Risque") -> str:
    return RISK_LEVEL_LABELS.get(str(level or "").lower(), fallback)


# ===========================================================================
# Store de demonstration (en memoire, partage entre sessions)
# ===========================================================================


class DemoStore:
    """Comptes clients + transactions, charges depuis ``demo_data.json``."""

    def __init__(self, users: List[Dict[str, Any]], transactions: List[Dict[str, Any]]) -> None:
        self.users: Dict[str, Dict[str, Any]] = {u["user_id"]: dict(u) for u in users}
        self.transactions: List[Dict[str, Any]] = [dict(t) for t in transactions]

    # ------------------------------------------------------------------ users
    def all_users(self) -> List[Dict[str, Any]]:
        return [self.users[key] for key in sorted(self.users)]

    def user(self, user_id: str) -> Dict[str, Any]:
        return self.users[user_id]

    def credit(self, user_id: str, delta: float) -> float:
        self.users[user_id]["balance"] = round(float(self.users[user_id]["balance"]) + delta, 2)
        return self.users[user_id]["balance"]

    # ----------------------------------------------------------- transactions
    def txs(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        rows = [t for t in self.transactions if t["user_id"] == user_id]
        rows.sort(key=lambda t: t["created_at"], reverse=True)
        return rows[:limit] if limit else rows

    def add(self, tx: Dict[str, Any]) -> None:
        self.transactions.append(tx)

    def tx_by_id(self, tx_id: str) -> Optional[Dict[str, Any]]:
        for tx in self.transactions:
            if tx["id"] == tx_id:
                return tx
        return None

    # ------------------------------------------------------------- indicateurs
    def velocity_1h(self, user_id: str) -> int:
        cutoff = (datetime.now() - timedelta(hours=1)).isoformat(timespec="seconds")
        return sum(1 for t in self.txs(user_id) if t["created_at"] >= cutoff)

    def risk_history(self, user_id: str, limit: int = 12) -> float:
        rows = self.txs(user_id, limit)
        if not rows:
            return 0.05
        weighted, weights = 0.0, 0.0
        for i, tx in enumerate(rows):
            w = 0.85 ** i
            weighted += (float(tx["risk_score"] or 0.0) / 100.0) * w
            weights += w
        return round(weighted / max(weights, 1e-6), 3)

    def security(self, user_id: str, limit: int = 25) -> Dict[str, Any]:
        """Score dynamique 0-100 (meme formule que l'API)."""
        rows = self.txs(user_id, limit)
        if not rows:
            return {"score": 92.0, "level": "excellent", "incidents": 0, "avg_risk": 0.0}
        weighted, weights, incidents = 0.0, 0.0, 0
        for i, tx in enumerate(rows):
            w = 0.92 ** i
            weighted += float(tx["risk_score"] or 0.0) * w
            weights += w
            if float(tx["risk_score"] or 0.0) >= 70:
                incidents += 1
        avg_risk = weighted / max(weights, 1e-6)
        score = round(max(35.0, min(99.4, 100.0 - avg_risk * 0.72 - incidents * 3.2)), 1)
        level = "excellent" if score >= 85 else "bon" if score >= 70 else "moyen" if score >= 55 else "fragile"
        return {"score": score, "level": level, "incidents": incidents, "avg_risk": round(avg_risk, 1)}

    def alerts(self, user_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        rows = [t for t in self.txs(user_id)
                if float(t["risk_score"] or 0.0) >= 60 or t["status"] in ("declined", "expired")]
        return rows[:limit]

    # ------------------------------------------------------------- analytics
    def spend_by_category(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        buckets: Dict[str, Dict[str, Any]] = {}
        for tx in self.txs(user_id):
            if tx["created_at"] < cutoff or tx["status"] != "approved":
                continue
            bucket = buckets.setdefault(tx["merchant_type"], {"category": tx["merchant_type"], "amount": 0.0, "count": 0})
            bucket["amount"] = round(bucket["amount"] + float(tx["amount"]), 2)
            bucket["count"] += 1
        rows = sorted(buckets.values(), key=lambda b: b["amount"], reverse=True)
        for row in rows:
            row["label"] = merchant_label(row["category"])
        return rows

    def daily_trend(self, user_id: str, days: int = 14) -> List[Dict[str, Any]]:
        series: List[Dict[str, Any]] = []
        txs = self.txs(user_id)
        for offset in range(days - 1, -1, -1):
            day = (datetime.now() - timedelta(days=offset)).date().isoformat()
            day_txs = [t for t in txs if (t["created_at"] or "")[:10] == day]
            approved = [t for t in day_txs if t["status"] in ("approved", "pending_auth")]
            series.append({
                "date": day,
                "amount": round(sum(float(t["amount"]) for t in approved), 2),
                "anomalies": sum(1 for t in day_txs if float(t["risk_score"] or 0) >= 60),
                "risk_sum": sum(float(t["risk_score"] or 0) for t in day_txs),
                "count": len(day_txs),
            })
        return series

    def hourly_activity(self, user_id: str) -> List[int]:
        hours = [0] * 24
        for tx in self.txs(user_id):
            hours[int(tx["hour"] or 0) % 24] += 1
        return hours


# ===========================================================================
# Chargement (moteur IA + donnees)
# ===========================================================================


@st.cache_resource(show_spinner="🧠 Chargement du moteur IA (RandomForest + IsolationForest)…")
def get_engine() -> FraudEngine:
    return FraudEngine()


def _load_store() -> DemoStore:
    """Charge la fixture ``demo_data.json`` dans un store independant.

    Repli minimal si la fixture n'a pas ete generee (python export_demo.py).
    """
    if DATA_FILE.exists():
        payload = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        return DemoStore(payload["users"], payload["transactions"])
    users = [
        {"user_id": "user_001", "name": "Awa Diallo", "email": "awa.diallo@novapay.io",
         "phone": "+33 6 12 •• •• 42", "home_city": "Paris, FR", "avg_amount": 46.0,
         "monthly_income": 3250.0, "balance": 4185.75, "currency": "EUR",
         "card_last4": "4821", "accent": "#8b5cf6"},
    ]
    return DemoStore(users, [])


def get_store() -> DemoStore:
    """Store propre a la session courante.

    Chaque visiteur a sa propre copie des donnees de demonstration : les
    paiements simules n'affectent donc pas les autres sessions.
    """
    session = st.session_state
    if "store" not in session:
        session["store"] = _load_store()
    return session["store"]


def reset_store() -> DemoStore:
    st.session_state["store"] = _load_store()
    return st.session_state["store"]


# ===========================================================================
# Pipeline de decision (score IA -> MFA -> code d'autorisation)
# ===========================================================================


def score_transaction(store: DemoStore, engine: FraudEngine, payload: Dict[str, Any],
                      user: Dict[str, Any]) -> Dict[str, Any]:
    """Score IA + explications (sans aucune ecriture)."""
    amount_eur = to_base(payload["amount"], payload.get("currency") or BASE_CURRENCY)
    location = payload.get("location") or resolve_location(payload.get("country"), user["home_city"])
    is_new_device = bool(payload.get("is_new_device"))
    tx_input = {
        "amount": amount_eur,
        "hour": int(payload.get("hour") if payload.get("hour") is not None else datetime.now().hour),
        "location": location,
        "merchant_type": payload.get("merchant_type") or "ecommerce",
        "is_new_device": is_new_device,
        "tx_velocity_1h": store.velocity_1h(user["user_id"]),
        "is_weekend": int(datetime.now().weekday() >= 5),
    }
    profile = {
        "home_city": user["home_city"],
        "avg_amount": float(user["avg_amount"]),
        "risk_history": store.risk_history(user["user_id"]),
    }
    verdict = engine.predict(tx_input, profile)
    verdict["amount_eur"] = amount_eur
    verdict["location"] = location
    return verdict


def process_payment(store: DemoStore, engine: FraudEngine, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Applique la politique de decision puis enregistre la transaction."""
    user = store.user(payload["user_id"])
    display = payload.get("currency") or BASE_CURRENCY
    verdict = score_transaction(store, engine, payload, user)
    risk = float(verdict["risk_score"])
    amount_eur = verdict["amount_eur"]

    tx: Dict[str, Any] = {
        "id": new_id("tx", 12),
        "user_id": user["user_id"],
        "amount": round(amount_eur, 2),
        "currency": BASE_CURRENCY,
        "merchant": payload.get("merchant") or random.choice(MERCHANTS_BY_CATEGORY.get(payload["merchant_type"], ["Marchand"])),
        "merchant_type": payload["merchant_type"],
        "location": verdict["location"],
        "hour": int(payload.get("hour") if payload.get("hour") is not None else datetime.now().hour),
        "channel": payload.get("channel") or "mobile_app",
        "device_id": payload.get("device_id") or f"dev_{user['user_id']}_principal",
        "is_new_device": int(bool(payload.get("is_new_device"))),
        "risk_score": risk,
        "risk_level": verdict["risk_level"],
        "decision": verdict["decision"],
        "status": "approved",
        "factors": localize_factors(verdict["factors"], display),
        "components": verdict["components"],
        "auth_code": None,
        "is_anomaly": bool(risk >= 60),
        "created_at": now_iso(),
    }

    result: Dict[str, Any] = {
        "transaction": tx, "risk_score": risk, "risk_level": verdict["risk_level"],
        "risk_label": verdict["risk_label"], "factors": tx["factors"],
        "currency": display, "authorization_code": None, "otp_challenge": None,
        "balance_after": None, "message": "",
    }

    if risk >= AUTO_BLOCK_THRESHOLD:
        tx["status"], tx["decision"] = "declined", "BLOCKED"
        result["message"] = "Transaction bloquée automatiquement par l'IA (risque critique)."
    elif risk >= RISK_CHALLENGE_THRESHOLD:
        tx["status"] = "pending_auth"
        code = new_auth_code()
        result["otp_challenge"] = {
            "code": code, "expires_at": time.time() + OTP_TTL_SECONDS,
            "attempts": 0, "masked_phone": user["phone"],
        }
        result["message"] = (f"Risque {risk:.0f} % ≥ {RISK_CHALLENGE_THRESHOLD:.0f} % : "
                             "authentification dynamique requise.")
    else:
        code = new_auth_code()
        tx["auth_code"] = code
        result["authorization_code"] = code
        result["balance_after"] = store.credit(user["user_id"], -tx["amount"])
        result["message"] = f"Risque {risk:.0f} % < {RISK_CHALLENGE_THRESHOLD:.0f} % : paiement validé automatiquement."

    store.add(tx)
    result["transaction"] = tx
    result["status"] = tx["status"]
    return result


def verify_otp(store: DemoStore, challenge: Dict[str, Any], code_input: str) -> Dict[str, Any]:
    """Valide (ou refuse) le code MFA d'une transaction en attente."""
    tx = store.tx_by_id(challenge["tx_id"])
    if tx is None:
        return {"ok": False, "error": "Transaction introuvable."}
    if tx["status"] != "pending_auth":
        return {"ok": False, "error": f"Transaction déjà traitée (statut : {status_label(tx['status'])})."}
    if time.time() > float(challenge["expires_at"]):
        tx["status"], tx["decision"] = "expired", "EXPIRED"
        return {"ok": False, "error": "Code expiré — relancez la transaction."}

    if normalize_otp(code_input) != normalize_otp(challenge["code"]):
        challenge["attempts"] += 1
        remaining = OTP_MAX_ATTEMPTS - challenge["attempts"]
        if remaining <= 0:
            tx["status"], tx["decision"] = "declined", "BLOCKED"
            return {"ok": False, "locked": True, "error": "Trop de tentatives : transaction bloquée par sécurité."}
        return {"ok": False, "error": f"Code MFA invalide. Tentatives restantes : {remaining}."}

    tx["status"], tx["decision"], tx["auth_code"] = "approved", "ACCEPTED_AFTER_OTP", challenge["code"]
    balance_after = store.credit(tx["user_id"], -float(tx["amount"]))
    return {
        "ok": True,
        "transaction": tx,
        "authorization_code": challenge["code"],
        "balance_after": balance_after,
        "message": "Code MFA validé : authentification dynamique réussie, paiement autorisé.",
    }


# ---------------------------------------------------------------------------
# Callbacks MFA
# Les callbacks s'executent *avant* la re-execution du script : ils peuvent
# donc modifier librement les cles de widgets (st.session_state.otp_input, ...).
# ---------------------------------------------------------------------------


def _otp_fill() -> None:
    challenge = st.session_state.get("challenge") or {}
    st.session_state["otp_input"] = normalize_otp(challenge.get("code", ""))
    st.session_state.pop("otp_error", None)


def _otp_validate() -> None:
    session = st.session_state
    challenge = session.get("challenge")
    if not challenge:
        return
    outcome = verify_otp(get_store(), challenge, session.get("otp_input") or "")
    session["otp_input"] = ""
    if outcome["ok"]:
        session["challenge"] = None
        session["receipt"] = outcome
        session.pop("otp_error", None)
    else:
        session["otp_error"] = outcome["error"]
        if outcome.get("locked"):
            session["challenge"] = None


def _otp_resend() -> None:
    session = st.session_state
    challenge = session.get("challenge")
    if not challenge:
        return
    challenge["code"] = new_auth_code()
    challenge["expires_at"] = time.time() + OTP_TTL_SECONDS
    challenge["attempts"] = 0
    session["otp_input"] = ""
    session.pop("otp_error", None)


def _otp_refuse() -> None:
    session = st.session_state
    challenge = session.get("challenge")
    tx = get_store().tx_by_id(challenge["tx_id"]) if challenge else None
    if tx is not None and tx["status"] == "pending_auth":
        tx["status"], tx["decision"] = "declined", "REFUSED_BY_CUSTOMER"
    session["challenge"] = None
    session["otp_input"] = ""
    session.pop("otp_error", None)


# ===========================================================================
# Composants d'interface
# ===========================================================================

CSS = """
<style>
  .block-container { padding-top: 2.2rem; }
  .nova-code {
      font-family: 'JetBrains Mono', ui-monospace, monospace; letter-spacing: .18em;
      background: linear-gradient(135deg, rgba(139,92,246,.28), rgba(34,211,238,.18));
      border: 1px dashed rgba(167,139,250,.6); border-radius: 16px;
      padding: 14px 18px; text-align: center; margin-top: 6px;
  }
  .nova-code .label { font-size: .68rem; text-transform: uppercase; letter-spacing: .18em; color: #c4b5fd; }
  .nova-code .value { font-size: 2rem; font-weight: 800; color: #ffffff; margin-top: 4px; }
  .nova-reason {
      background: rgba(244,63,94,.10); border: 1px solid rgba(244,63,94,.28);
      border-radius: 12px; padding: 10px 14px; margin-top: 8px; font-size: .85rem;
  }
  .nova-reason .title { font-size: .65rem; text-transform: uppercase; letter-spacing: .16em; color: #fda4af; }
  div[data-testid="stMetricValue"] { font-weight: 800; }
</style>
"""


def apply_theme() -> None:
    st.set_page_config(
        page_title="NovaPay AI — Démo Streamlit",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 290) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=34, b=8),
        height=height,
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, x=0),
    )
    return fig


def code_chip(code: str, label: str = "Code de sécurité à usage unique (généré par l'IA)") -> str:
    return (f'<div class="nova-code"><div class="label">{label}</div>'
            f'<div class="value">{code}</div></div>')


def show_alert_rows(store: DemoStore, user_id: str, display_currency: str, limit: int = 3) -> None:
    alerts = store.alerts(user_id, limit)
    if not alerts:
        st.info("Aucune tentative de fraude détectée sur ce compte. 🎉")
        return
    for index, alert in enumerate(alerts, start=1):
        factors = localize_factors(alert["factors"] or [], display_currency)
        reason = " · ".join(factors[:2]) if factors else "Comportement atypique détecté par l'IA"
        created = datetime.fromisoformat(alert["created_at"])
        if alert["status"] == "declined":
            action = "🚫 Transaction bloquée" if alert["decision"] == "BLOCKED" else "🚫 Refusée par le client"
        elif alert["status"] == "expired":
            action = "⌛ Code MFA expiré — transaction abandonnée"
        else:
            action = "🔐 Validée après authentification MFA (code à usage unique)"
        with st.container(border=True):
            head_left, head_right = st.columns([4, 1])
            head_left.markdown(
                f"**Alerte #{index} — {alert['merchant']}** &nbsp; "
                + risk_badge(float(alert["risk_score"]), status_label("approved") if alert["status"] == "approved" else "Tentative"),
                unsafe_allow_html=True,
            )
            head_right.caption(merchant_label(alert["merchant_type"]))
            info_left, info_mid, info_right = st.columns(3)
            info_left.markdown(f"🕐 **Heure**  \n{created.strftime('%d/%m/%Y à %H:%M')} ({int(alert['hour']):02d}h)")
            info_mid.markdown(f"💶 **Montant**  \n{fmt_money(to_display(float(alert['amount']), display_currency), display_currency)}")
            info_right.markdown(f"🌍 **Lieu**  \n{alert['location']}")
            st.markdown(
                f'<div class="nova-reason"><div class="title">Motif du blocage</div>{reason}</div>',
                unsafe_allow_html=True,
            )
            st.caption(action + f" · score de risque {float(alert['risk_score']):.0f} %")


def security_gauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 34, "color": "#e2e8f0"}, "suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)"},
            "bar": {"color": risk_color(100 - score), "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,.06)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 55], "color": "rgba(244,63,94,.25)"},
                {"range": [55, 70], "color": "rgba(251,191,36,.18)"},
                {"range": [70, 85], "color": "rgba(52,211,153,.16)"},
                {"range": [85, 100], "color": "rgba(52,211,153,.28)"},
            ],
        },
    ))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=14, r=14, t=10, b=0), height=170)
    return fig


# ===========================================================================
# Onglets
# ===========================================================================


def render_dashboard_tab(store: DemoStore, engine: FraudEngine, user: Dict[str, Any], currency: str) -> None:
    chart_left, chart_right = st.columns(2)

    # --- Depenses par categorie
    with chart_left:
        st.subheader("Dépenses par catégorie")
        rows = store.spend_by_category(user["user_id"])
        if rows:
            fig = go.Figure(go.Pie(
                labels=[row["label"] for row in rows[:8]],
                values=[to_display(row["amount"], currency) for row in rows[:8]],
                hole=0.58,
                marker=dict(colors=CAT_CHART_COLORS, line=dict(color="rgba(5,7,15,.9)", width=2)),
                textinfo="percent",
                hovertemplate="%{label}<br>%{value:.2f} " + currency + "<extra></extra>",
            ))
            st.plotly_chart(style_fig(fig, 320), width="stretch")
            st.caption(f"30 derniers jours · montants en {currency}")
        else:
            st.info("Aucune dépense sur les 30 derniers jours.")

    # --- Activite par heure
    with chart_right:
        st.subheader("Activité par heure")
        hours = store.hourly_activity(user["user_id"])
        colors = ["rgba(244,63,94,.75)" if (h >= 23 or h <= 5) else "rgba(139,92,246,.75)" for h in range(24)]
        fig = go.Figure(go.Bar(x=[f"{h}h" for h in range(24)], y=hours, marker_color=colors))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(gridcolor="rgba(255,255,255,.06)", title=None))
        st.plotly_chart(style_fig(fig, 320), width="stretch")
        st.caption("Plages sensibles 23h–5h surlignées en rouge")

    # --- Tendance 14 jours + anomalies
    st.subheader("Analyse comportementale — 14 jours")
    trend = store.daily_trend(user["user_id"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[row["date"] for row in trend], y=[to_display(row["amount"], currency) for row in trend],
        mode="lines+markers", name="Dépenses", line=dict(color="#a78bfa", width=2.5),
        fill="tozeroy", fillcolor="rgba(139,92,246,.22)",
        hovertemplate="%{x}<br>%{y:.2f} " + currency + "<extra></extra>",
    ))
    anomalies = [row for row in trend if row["anomalies"]]
    if anomalies:
        fig.add_trace(go.Scatter(
            x=[row["date"] for row in anomalies], y=[to_display(row["amount"], currency) for row in anomalies],
            mode="markers", name="Anomalies IA",
            marker=dict(color="#f43f5e", size=13, line=dict(color="white", width=2)),
            hovertemplate="%{x}<br>%{customdata} anomalie(s) IA<extra></extra>",
            customdata=[row["anomalies"] for row in anomalies],
        ))
    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(gridcolor="rgba(255,255,255,.06)"))
    st.plotly_chart(style_fig(fig, 320), width="stretch")
    st.caption(f"Montants en {currency} · les points rouges signalent les jours avec au moins une anomalie IA")

    # --- Transactions
    st.subheader("Dernières transactions")
    txs = store.txs(user["user_id"], 20)
    if not txs:
        st.info("Aucune transaction pour l'instant — lancez une simulation dans l'onglet « Simuler un paiement ».")
        return
    table = pd.DataFrame([
        {
            "Date": datetime.fromisoformat(tx["created_at"]).strftime("%d/%m %H:%M"),
            "Marchand": tx["merchant"],
            "Catégorie": merchant_label(tx["merchant_type"]),
            "Lieu": tx["location"],
            "Montant": fmt_money(to_display(float(tx["amount"]), currency), currency),
            "Risque": round(float(tx["risk_score"]), 1),
            "Statut": status_label(tx["status"]),
            "Code d'autorisation": tx.get("auth_code") or "—",
        }
        for tx in txs
    ])
    styled = table.style.map(lambda value: f"color:{risk_color(float(value))};font-weight:700", subset=["Risque"])
    st.dataframe(styled, width="stretch", hide_index=True)
    st.caption("Cliquez sur un en-tête pour trier · le code d'autorisation n'est généré que pour les paiements validés.")

    # --- Detail d'une transaction
    with st.expander("🔎 Détail d'une transaction (score, facteurs, décomposition)"):
        options = {
            f"#{index + 1} · {tx['merchant']} — {datetime.fromisoformat(tx['created_at']).strftime('%d/%m %H:%M')} "
            f"({fmt_money(to_display(float(tx['amount']), currency), currency)})": tx
            for index, tx in enumerate(txs)
        }
        chosen = st.selectbox("Transaction", list(options), key="detail_tx")
        tx = options[chosen]
        comp = tx.get("components") or {}
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Score de risque", f"{float(tx['risk_score']):.0f} %")
        col_b.metric("Statut", status_label(tx["status"]))
        col_c.metric("Décision", tx["decision"])
        if tx.get("auth_code"):
            st.markdown(code_chip(tx["auth_code"]), unsafe_allow_html=True)
        for key, label in (("random_forest", "RandomForest"), ("isolation_forest", "IsolationForest"),
                           ("rules", "Règles métier"), ("business_rules", "Règles métier")):
            if key in comp:
                value = float(comp[key])
                if value <= 1.0:          # les composantes brutes sont dans [0, 1]
                    value *= 100.0
                st.progress(min(1.0, value / 100.0), text=f"{label} : {value:.1f} %")
        st.markdown("**Facteurs analysés par l'IA**")
        for factor in localize_factors(tx.get("factors") or [], currency):
            st.markdown(f"- {factor}")


def render_payment_tab(store: DemoStore, engine: FraudEngine, user: Dict[str, Any], currency: str) -> None:
    symbol = currency_info(currency)["symbol"]
    category_keys = list(MERCHANT_CATEGORIES.keys())
    country_codes = [item["code"] for item in COUNTRIES]
    country_names = {item["code"]: f"{item['name']} · risque {item['risk_hint'] * 100:.0f} %" for item in COUNTRIES}

    # --- Initialisation des champs du simulateur (valeurs par defaut propres au client)
    session = st.session_state
    if session.get("sim_user") != user["user_id"]:
        session["sim_user"] = user["user_id"]
        session.pop("sim_amount", None)
        session.pop("sim_merchant", None)
    session.setdefault("sim_amount", round(float(user["avg_amount"]) * 0.9, 2))
    session.setdefault("sim_merchant", "")
    session.setdefault("sim_category", "alimentation")
    session.setdefault("sim_country", "FR")
    session.setdefault("sim_hour", datetime.now().hour)
    session.setdefault("sim_device", False)
    session.setdefault("sim_channel", "mobile_app")
    if session["sim_category"] not in category_keys:
        session["sim_category"] = "alimentation"
    if session["sim_country"] not in country_codes:
        session["sim_country"] = "FR"

    # --- Bandeau des transactions en attente de MFA
    pending = [tx for tx in store.txs(user["user_id"]) if tx["status"] == "pending_auth"]
    if pending and not st.session_state.get("challenge"):
        tx = pending[0]
        st.warning(f"⏳ Une transaction attend une authentification : {tx['merchant']} — "
                   f"{fmt_money(to_display(float(tx['amount']), currency), currency)}.")
        if st.button("Recevoir un code MFA", key="resume_challenge"):
            code = new_auth_code()
            st.session_state["challenge"] = {
                "tx_id": tx["id"], "code": code, "expires_at": time.time() + OTP_TTL_SECONDS,
                "attempts": 0, "masked_phone": user["phone"],
            }
            st.session_state["receipt"] = None
            st.toast("Nouveau code de sécurité envoyé (simulation SMS).", icon="📩")
            st.rerun()

    # ============================================================== Recu final
    receipt = st.session_state.get("receipt")
    if receipt:
        tx = receipt["transaction"]
        with st.container(border=True):
            st.success("✅ Paiement autorisé — transaction débitée.")
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Montant", fmt_money(to_display(float(tx["amount"]), currency), currency))
            col_b.metric("Marchand", tx["merchant"])
            col_c.metric("Score de risque IA", f"{float(tx['risk_score']):.0f} %")
            col_d.metric("Nouveau solde",
                         fmt_money(to_display(float(store.user(user["user_id"])["balance"]), currency), currency))
            st.caption(f"{merchant_label(tx['merchant_type'])} · {tx['location']} · "
                       f"{datetime.fromisoformat(tx['created_at']).strftime('%d/%m/%Y à %H:%M')} · "
                       f"{'validé après MFA' if tx['decision'] == 'ACCEPTED_AFTER_OTP' else 'validé automatiquement (risque faible)'}")
            if receipt.get("authorization_code"):
                st.markdown(code_chip(receipt["authorization_code"]), unsafe_allow_html=True)
                st.caption("Ce code à usage unique a autorisé le paiement : il n'est plus réutilisable.")
        if st.button("🔄 Nouvelle simulation", key="receipt_new"):
            st.session_state["receipt"] = None
            st.rerun()
        return

    # ============================================================== Defi MFA
    challenge = st.session_state.get("challenge")
    if challenge:
        tx = store.tx_by_id(challenge["tx_id"])
        if tx is None or tx["status"] != "pending_auth":
            st.session_state["challenge"] = None
            st.rerun()
        with st.container(border=True):
            st.subheader("🔐 Authentification dynamique (MFA)")
            st.caption("L'IA a détecté un risque élevé : saisissez le code à usage unique envoyé par SMS.")
            st.markdown(
                f"Transaction de **{fmt_money(to_display(float(tx['amount']), currency), currency)}** "
                f"chez **{tx['merchant']}** à {tx['location']} — "
                f"score de risque **{float(tx['risk_score']):.0f} %**."
            )
            sms_col, fill_col = st.columns([3, 1])
            sms_col.info(f"📩 **Simulation SMS** — code reçu : `{challenge['code']}` "
                         f"(destinataire {challenge['masked_phone']})", icon="📱")
            fill_col.button("Remplir", key="otp_fill", width="stretch", on_click=_otp_fill)

            if hasattr(st, "fragment"):
                @st.fragment(run_every=1.0)
                def countdown(expires_at: float = challenge["expires_at"]) -> None:
                    remaining = max(0, int(expires_at - time.time()))
                    if remaining > 0:
                        st.progress(min(1.0, remaining / OTP_TTL_SECONDS),
                                    text=f"⏳ Code valable encore {remaining // 60:02d}:{remaining % 60:02d}")
                    else:
                        st.error("⌛ Code expiré — refusez puis relancez la transaction.")
                countdown()
            else:
                remaining = max(0, int(challenge["expires_at"] - time.time()))
                st.progress(min(1.0, remaining / OTP_TTL_SECONDS),
                            text=f"⏳ Code valable encore {remaining // 60:02d}:{remaining % 60:02d}")

            if st.session_state.get("otp_error"):
                st.error(st.session_state["otp_error"])

            st.text_input("Code MFA (format OTP-1234)", key="otp_input",
                          placeholder="1234", max_chars=8,
                          help="Saisissez les 4 chiffres ; le préfixe OTP- est facultatif.")
            st.caption(f"Tentatives : {challenge['attempts']}/{OTP_MAX_ATTEMPTS}")

            btn_validate, btn_resend, btn_refuse = st.columns(3)
            btn_validate.button("✅ Valider", type="primary", width="stretch",
                                key="otp_validate", on_click=_otp_validate)
            btn_resend.button("📨 Renvoyer", width="stretch", key="otp_resend", on_click=_otp_resend)
            btn_refuse.button("🚫 Refuser", width="stretch", key="otp_refuse", on_click=_otp_refuse)
        return

    # ============================================================ Formulaire
    st.caption("Le **score de risque est calculé en temps réel** : chaque modification des champs "
               "met à jour l'analyse IA instantanément.")

    quick_left, quick_right, quick_hint = st.columns([1, 1, 2])
    if quick_left.button("🟢 Scénario normal", width="stretch"):
        category = random.choice(NORMAL_CATEGORY_POOL)
        st.session_state.update({
            "sim_amount": round(float(user["avg_amount"]) * random.uniform(0.6, 1.2), 2),
            "sim_merchant": random.choice(MERCHANTS_BY_CATEGORY[category]),
            "sim_category": category,
            "sim_country": "FR",
            "sim_hour": 13,
            "sim_device": False,
        })
        st.rerun()
    if quick_right.button("🔴 Scénario suspect", width="stretch"):
        category = random.choice(SUSPICIOUS_CATEGORY_POOL)
        st.session_state.update({
            "sim_amount": round(float(user["avg_amount"]) * random.uniform(6.0, 12.0), 2),
            "sim_merchant": random.choice(MERCHANTS_BY_CATEGORY[category]),
            "sim_category": category,
            "sim_country": random.choice(["RU", "PA", "HK", "NG", "AE"]),
            "sim_hour": random.choice([1, 2, 3, 4, 23]),
            "sim_device": True,
        })
        st.rerun()
    quick_hint.caption("Les scénarios pré-remplissent le formulaire avec un cas typique (normal ou frauduleux).")

    form_col, preview_col = st.columns([3, 2], gap="large")

    with form_col:
        amount = st.number_input(f"Montant ({symbol})", min_value=0.5, max_value=10_000_000.0,
                                 step=1.0, format="%.2f", key="sim_amount")

        merchant = st.text_input("Marchand", key="sim_merchant",
                                 placeholder="ex : Starbucks, Binance…",
                                 help="Saisissez librement le nom du marchand.")

        category = st.selectbox(
            "Catégorie de marchand",
            category_keys,
            format_func=lambda key: f"{MERCHANT_CATEGORIES[key]['label']} — risque {MERCHANT_CATEGORIES[key]['risk'] * 100:.0f} %",
            key="sim_category",
        )
        st.caption("Suggestions : " + ", ".join(MERCHANTS_BY_CATEGORY.get(category, [])[:4]))

        country = st.selectbox("Pays de la transaction", country_codes,
                               format_func=lambda code: country_names[code], key="sim_country")

        hour = st.slider("Heure de la transaction", 0, 23, key="sim_hour")
        device_col, channel_col = st.columns(2)
        is_new_device = device_col.checkbox("Appareil non reconnu", key="sim_device")
        channel = channel_col.selectbox("Canal", ["mobile_app", "web", "atm", "api"], key="sim_channel",
                                        format_func=lambda value: {"mobile_app": "Application mobile", "web": "Web",
                                                                   "atm": "Distributeur", "api": "API"}[value])

    # --------------------------------------------------- Score temps reel
    payload = {
        "user_id": user["user_id"], "amount": float(amount), "currency": currency,
        "merchant": merchant.strip(), "merchant_type": category, "country": country,
        "hour": int(hour), "is_new_device": bool(is_new_device), "channel": channel,
    }
    verdict = score_transaction(store, engine, payload, user)
    risk = float(verdict["risk_score"])

    with preview_col:
        with st.container(border=True):
            st.markdown("##### 🧠 Score de risque IA — *temps réel*")
            st.metric("Risque estimé", f"{risk:.0f} %")
            st.progress(min(1.0, risk / 100.0))
            st.markdown(risk_badge(risk, risk_level_label(verdict["risk_level"], verdict["risk_label"])), unsafe_allow_html=True)
            if risk >= AUTO_BLOCK_THRESHOLD:
                st.error("🚫 Blocage automatique (risque critique).")
            elif risk >= RISK_CHALLENGE_THRESHOLD:
                st.warning("🔐 Authentification dynamique (code OTP-XXXX) requise.")
            else:
                st.success("✅ Validation automatique (risque faible).")
            st.caption(f"Ville analysée : {verdict['location']} · seuil MFA : {RISK_CHALLENGE_THRESHOLD:.0f} % · "
                       f"montant équivalent {fmt_money(verdict['amount_eur'], 'EUR')}")
            st.markdown("**Facteurs détectés**")
            for factor in verdict["factors"][:4]:
                st.markdown(f"- {factor}")

    if st.button("💳 Analyser & payer", type="primary", width="stretch", key="submit_payment"):
        result = process_payment(store, engine, payload)
        if result["status"] == "approved":
            st.session_state["receipt"] = result
            st.toast(f"Paiement validé (risque {result['risk_score']:.0f} %).", icon="✅")
            st.balloons()
        elif result["status"] == "pending_auth":
            st.session_state["challenge"] = {
                "tx_id": result["transaction"]["id"],
                "code": result["otp_challenge"]["code"],
                "expires_at": result["otp_challenge"]["expires_at"],
                "attempts": 0,
                "masked_phone": result["otp_challenge"]["masked_phone"],
            }
            st.toast(f"Risque {result['risk_score']:.0f} % : authentification MFA déclenchée.", icon="🔐")
        else:
            st.error(result["message"])
        st.rerun()


def render_alerts_tab(store: DemoStore, user: Dict[str, Any], currency: str) -> None:
    st.subheader("🚨 Historique des alertes")
    st.caption("Tentatives de fraude détectées par l'IA — heure, montant et motif du blocage.")
    alerts = store.alerts(user["user_id"], 3)
    show_alert_rows(store, user["user_id"], currency, limit=3)
    total = len([t for t in store.txs(user["user_id"])
                 if float(t["risk_score"] or 0) >= 60 or t["status"] in ("declined", "expired")])
    if total > 3:
        with st.expander(f"Voir plus d'alertes ({total} au total)"):
            show_alert_rows(store, user["user_id"], currency, limit=min(total, 20))
    if alerts:
        st.caption("Rappel : toute transaction ≥ 30 % de risque exige une authentification dynamique "
                   "(code OTP-XXXX). Les tentatives bloquées n'ont donné lieu à aucun débit.")


def render_analytics_tab(store: DemoStore, user: Dict[str, Any], currency: str) -> None:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Distribution du risque (20 dernières transactions)")
        txs = store.txs(user["user_id"], 20)
        buckets = {"Faible (<30 %)": 0, "Modéré (30-59 %)": 0, "Élevé (60-84 %)": 0, "Critique (≥85 %)": 0}
        for tx in txs:
            score = float(tx["risk_score"] or 0)
            key = "Faible (<30 %)" if score < 30 else "Modéré (30-59 %)" if score < 60 else "Élevé (60-84 %)" if score < 85 else "Critique (≥85 %)"
            buckets[key] += 1
        fig = go.Figure(go.Bar(
            x=list(buckets.values()), y=list(buckets.keys()), orientation="h",
            marker_color=["#34d399", "#fbbf24", "#fb923c", "#f43f5e"],
            text=list(buckets.values()), textposition="auto",
        ))
        fig.update_layout(xaxis=dict(gridcolor="rgba(255,255,255,.06)"), yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig, 280), width="stretch")

    with col_right:
        st.subheader("Dépenses 30 jours par catégorie")
        rows = store.spend_by_category(user["user_id"])
        if rows:
            table = pd.DataFrame([
                {"Catégorie": row["label"], "Montant": fmt_money(to_display(row["amount"], currency), currency),
                 "Transactions": row["count"]}
                for row in rows
            ])
            st.dataframe(table, width="stretch", hide_index=True, height=280)
        else:
            st.info("Aucune dépense sur la période.")

    st.subheader("Rythme de dépense — 14 jours")
    trend = store.daily_trend(user["user_id"])
    table = pd.DataFrame([
        {"Date": row["date"],
         "Dépenses": fmt_money(to_display(row["amount"], currency), currency),
         "Risque moyen": round(row["risk_sum"] / row["count"], 1) if row["count"] else 0.0,
         "Anomalies IA": row["anomalies"]}
        for row in trend
    ])
    st.dataframe(table, width="stretch", hide_index=True)


def render_ai_tab(engine: FraudEngine, store: DemoStore, user: Dict[str, Any]) -> None:
    info = engine.info()
    metrics = info.get("metrics") or {}
    st.subheader("🤖 Moteur IA hybride")
    st.markdown(
        "Trois modèles combinés : **RandomForest** (classification supervisée, 240 arbres), "
        "**IsolationForest** (détection d'anomalies entraînée sur les seules transactions légitimes) "
        "et un **moteur de règles métier** (explicabilité + garde-fous).\n\n"
        "```\nscore = 100 × ( 0,46 × P(RandomForest) + 0,34 × P(IsolationForest) + 0,20 × Score de règles )^1,22\n```"
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.3f}")
    col_b.metric("Précision", f"{metrics.get('precision', 0):.3f}")
    col_c.metric("Rappel", f"{metrics.get('recall', 0):.3f}")
    col_d.metric("Seuil MFA", f"{info.get('threshold', RISK_CHALLENGE_THRESHOLD):.0f} %")
    trained_at = str(info.get("trained_at") or "")[:19]
    n_samples = metrics.get("n_samples_total")
    details = [f"modèle version {info.get('version', '—')}"]
    if n_samples:
        details.append(f"entraîné sur {int(n_samples):,}".replace(",", " ") + " transactions")
    if trained_at:
        details.append(f"dernier entraînement le {trained_at}")
    details.append(f"{info.get('n_features', 12)} variables")
    st.caption(" · ".join(details))

    st.subheader("Performances sur le jeu de test")
    performance = [
        {"Métrique": "AUC-ROC", "Valeur": f"{metrics.get('auc_roc', 0):.4f}"},
        {"Métrique": "Précision (fraudes)", "Valeur": f"{metrics.get('precision', 0):.4f}"},
        {"Métrique": "Rappel (fraudes)", "Valeur": f"{metrics.get('recall', 0):.4f}"},
        {"Métrique": "F1-score", "Valeur": f"{metrics.get('f1_score', 0):.4f}"},
        {"Métrique": "Score moyen — transactions légitimes", "Valeur": f"{metrics.get('mean_score_legit', 0):.1f} %"},
        {"Métrique": "Score moyen — transactions frauduleuses", "Valeur": f"{metrics.get('mean_score_fraud', 0):.1f} %"},
        {"Métrique": "Seuil d'authentification dynamique", "Valeur": f"{info.get('threshold', RISK_CHALLENGE_THRESHOLD):.0f} %"},
    ]
    st.dataframe(pd.DataFrame(performance), width="stretch", hide_index=True)

    st.subheader("Variables les plus déterminantes")
    importances = info.get("feature_importances") or {}
    if importances:
        top = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:8]
        fig = go.Figure(go.Bar(
            x=[value * 100 for _, value in top],
            y=[FEATURE_LABELS.get(key, key) for key, _ in top],
            orientation="h",
            marker_color="#8b5cf6",
            text=[f"{value * 100:.1f} %" for _, value in top], textposition="auto",
        ))
        fig.update_layout(xaxis=dict(gridcolor="rgba(255,255,255,.06)"), yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig, 330), width="stretch")

    st.subheader("Seuils de décision")
    st.dataframe(pd.DataFrame([
        {"Score de risque": "< 30 %", "Niveau": "Faible", "Décision": "✅ Validation automatique + code d'autorisation"},
        {"Score de risque": "30 – 59 %", "Niveau": "Modéré", "Décision": "🔐 Authentification dynamique (OTP-XXXX)"},
        {"Score de risque": "60 – 84 %", "Niveau": "Élevé", "Décision": "🔐 OTP + signalement dans les alertes"},
        {"Score de risque": "≥ 85 %", "Niveau": "Critique", "Décision": "🔐 OTP (blocage automatique activable)"},
    ]), width="stretch", hide_index=True)

    security = store.security(user["user_id"])
    st.caption(f"Pour **{user['name']}** : score de sécurité actuel {security['score']}/100 "
               f"({security['level']}) · risque moyen pondéré {security['avg_risk']} % "
               f"· {security['incidents']} incident(s) récent(s).")


# ===========================================================================
# Application
# ===========================================================================


def main() -> None:
    apply_theme()

    engine = get_engine()
    store = get_store()

    if not DATA_FILE.exists():
        st.warning("⚠️ Fixture `demo_data.json` introuvable — lancez `python export_demo.py` "
                   "pour générer l'historique de démonstration.", icon="📦")

    # ---------------------------------------------------------------- session
    session = st.session_state
    session.setdefault("user_select", "user_001")
    session.setdefault("currency_select", "EUR")
    session.setdefault("challenge", None)
    session.setdefault("receipt", None)

    users = store.all_users()
    user_ids = [item["user_id"] for item in users]
    if session["user_select"] not in user_ids:
        session["user_select"] = user_ids[0]
    currency_codes = [item["code"] for item in CURRENCIES]
    if session["currency_select"] not in currency_codes:
        session["currency_select"] = BASE_CURRENCY

    # ---------------------------------------------------------------- sidebar
    with st.sidebar:
        st.markdown("## 🛡️ NovaPay AI")
        st.caption("Prototype FinTech — détection de fraude IA & MFA dynamique")

        user_id = st.selectbox(
            "Client", user_ids,
            format_func=lambda uid: f"{store.user(uid)['name']} · {store.user(uid)['home_city'].split(',')[0]}",
            key="user_select",
        )
        user = store.user(user_id)

        currency = st.selectbox(
            "Devise d'affichage", currency_codes,
            format_func=lambda code: f"{currency_info(code)['symbol']} {code} — {currency_info(code)['label']}",
            key="currency_select",
        )
        if currency != BASE_CURRENCY:
            st.caption(f"1 € = {currency_info(currency)['rate']:.2f} {currency} (taux fixe de démonstration)")

        security = store.security(user["user_id"])
        st.markdown("### Score de sécurité")
        st.plotly_chart(security_gauge(security["score"]), width="stretch")
        level_labels = {"excellent": "Excellent", "bon": "Bon", "moyen": "Moyen", "fragile": "Fragile"}
        st.caption(f"**{level_labels.get(security['level'], security['level'])}** · "
                   f"{security['incidents']} incident(s) récent(s) · risque moyen {security['avg_risk']} %")

        st.divider()
        st.markdown("### Moteur IA")
        info = engine.info()
        st.caption(f"AUC-ROC **{info.get('metrics', {}).get('auc_roc', float('nan')):.3f}** · "
                   f"seuil MFA **{RISK_CHALLENGE_THRESHOLD:.0f} %** · "
                   f"RandomForest + IsolationForest + règles")

        st.divider()
        if st.button("♻️ Réinitialiser la démonstration", width="stretch"):
            reset_store()
            session["challenge"] = None
            session["receipt"] = None
            st.toast("Données de démonstration rechargées.", icon="♻️")
            st.rerun()
        st.caption("Les données sont fictives (fixture `demo_data.json`) — aucun paiement réel.")

    # ----------------------------------------------------------------- entete
    title_col, balance_col = st.columns([3, 2])
    with title_col:
        st.title("Tableau de bord")
        st.caption(f"Bonjour **{user['name']}** — {user['home_city']} · carte •••• {user['card_last4']} · "
                   f"{user['email']}")
    with balance_col:
        balance_display = to_display(float(user["balance"]), currency)
        st.metric("Solde disponible", fmt_money(balance_display, currency),
                  delta=f"≈ {fmt_money(float(user['balance']), 'EUR')}" if currency != BASE_CURRENCY else None,
                  delta_color="off")

    # ------------------------------------------------------------------- KPI
    txs_30d = [tx for tx in store.txs(user["user_id"])
               if tx["created_at"] >= (datetime.now() - timedelta(days=30)).isoformat(timespec="seconds")
               and tx["status"] == "approved"]
    spend_30d = sum(float(tx["amount"]) for tx in txs_30d)
    avg_risk = sum(float(tx["risk_score"] or 0) for tx in txs_30d) / len(txs_30d) if txs_30d else 0.0
    alerts_30d = sum(1 for tx in txs_30d if float(tx["risk_score"] or 0) >= 60)

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Dépenses 30 j", fmt_money(to_display(spend_30d, currency), currency),
                delta=f"{len(txs_30d)} tx", delta_color="off")
    kpi2.metric("Revenus 30 j", fmt_money(to_display(float(user["monthly_income"]), currency), currency))
    kpi3.metric("Ticket moyen",
                fmt_money(to_display(spend_30d / len(txs_30d), currency), currency) if txs_30d else "—")
    kpi4.metric("Risque moyen", f"{avg_risk:.1f} %")
    with kpi5:
        st.metric("Alertes", alerts_30d)
        if hasattr(st, "dialog"):
            if st.button("🔍 Détail", width="stretch", key="open_alerts"):
                st.session_state["show_alerts_dialog"] = True
        else:
            st.caption("Voir l'onglet 🚨 Alertes")
        st.caption("tentatives ≥ 60 %")

    # --------------------------------------------------- Boite de dialogue
    if session.get("show_alerts_dialog") and hasattr(st, "dialog"):
        @st.dialog("🚨 Historique des alertes", width="large")
        def alerts_dialog() -> None:
            st.caption("Les 3 dernières tentatives de fraude détectées par l'IA — heure, montant et motif du blocage.")
            show_alert_rows(store, user["user_id"], currency, limit=3)
        session["show_alerts_dialog"] = False
        alerts_dialog()

    st.divider()

    # ------------------------------------------------------------------- tabs
    tab_dash, tab_pay, tab_alerts, tab_analytics, tab_ai = st.tabs(
        ["📊 Tableau de bord", "💳 Simuler un paiement", "🚨 Alertes", "📈 Analytique", "🤖 Moteur IA"]
    )
    with tab_dash:
        render_dashboard_tab(store, engine, user, currency)
    with tab_pay:
        render_payment_tab(store, engine, user, currency)
    with tab_alerts:
        render_alerts_tab(store, user, currency)
    with tab_analytics:
        render_analytics_tab(store, user, currency)
    with tab_ai:
        render_ai_tab(engine, store, user)

    st.divider()
    st.caption("NovaPay AI · Proof of Concept — FastAPI + scikit-learn + Streamlit · "
               "données fictives, usage démonstration uniquement · "
               "version API complète (Swagger) : `uvicorn main:app`")


main()
