from __future__ import annotations
import uuid
import urllib.parse
import html
import pandas as pd
import streamlit as st
import numpy as np
import random
import psycopg2

from dictionaries import (
    var_groups,
    binary_vars,
    get_var_labels,
    get_var_units,
    get_var_descriptions,
    get_group_labels,
)

from experiment_config import (
    CONDITIONS,
    QUALTRICS_BASE_URL,
    SCENARIOS,
    xai_path,
    XAI_DIR,
)

# =========================================================
# Language texts (NL is leading, EN is parallel)
# =========================================================

TEXT = {
    "nl": {
        "language_title": "Taalkeuze",
        "language_text": "Kies alstublieft de taal waarin u het experiment wilt doorlopen.",
        "language_nl": "Nederlands",
        "language_en": "English",
        "step_counter": "Stap {current} van 4",

        "welcome_title": "Welkom",
        "consent_checkbox": "Ik heb de bovenstaande informatie gelezen en ga akkoord met deelname aan dit onderzoek.",
        "next": "Volgende",
        "back": "Terug",
        "predict": "Voorspellen",
        "continue": "Doorgaan naar vragenlijst",

        "system_explanation": "Uitleg van het systeem",
        "assigned_version": "Toegewezen versie",

        "scenario_header": "Scenario en voorspelling",
        "project_features": "#### Projectkenmerken",
        "prediction_header": "AI‑uitkomst",
        "explanation_header": "Uitleg",

        "no_explanation": "In deze versie van het systeem wordt geen uitleg bij de voorspelling gegeven.",

        "questionnaire": "Vragenlijst",
        "open_questionnaire": "Open vragenlijst",
        "copy_link": "Werkt de knop niet? Kopieer dan de onderstaande link en plak deze in uw browser.",
    },

    "en": {
        "language_title": "Language selection",
        "language_text": "Please select the language in which you would like to complete the experiment.",
        "language_nl": "Dutch",
        "language_en": "English",
        "step_counter": "Step {current} of 4",

        "welcome_title": "Welcome",
        "consent_checkbox": "I have read the information above and agree to participate in this study.",
        "next": "Next",
        "back": "Back",
        "predict": "Predict",
        "continue": "Continue to questionnaire",

        "system_explanation": "System explanation",
        "assigned_version": "Assigned version",

        "scenario_header": "Scenario and prediction",
        "project_features": "#### Project characteristics",
        "prediction_header": "AI output",
        "explanation_header": "Explanation",

        "no_explanation": "In this version of the system, no explanation is provided for the prediction.",

        "questionnaire": "Questionnaire",
        "open_questionnaire": "Open questionnaire",
        "copy_link": "If the button does not work, copy the link below and paste it into your browser.",
    },
}

def t(key: str, **kwargs) -> str:
    lang = st.session_state.get("language", "nl")
    txt = TEXT.get(lang, TEXT["nl"]).get(key, key)
    return txt.format(**kwargs) if kwargs else txt

def _init_participant_state():
    if "language" not in st.session_state:
        st.session_state.language = None
    if "pid" not in st.session_state:
        st.session_state.pid = str(uuid.uuid4())
    if "scenario_id" not in st.session_state:
        st.session_state.scenario_id = SCENARIOS[0].scenario_id
    if "exp_step" not in st.session_state:
        st.session_state.exp_step = 0

def _progress():
    if st.session_state.exp_step == 0:
        return
    st.progress((st.session_state.exp_step - 1) / 4)
    st.caption(t("step_counter", current=st.session_state.exp_step))

def _get_scenario():
    return next(s for s in SCENARIOS if s.scenario_id == st.session_state.scenario_id)

def _next():
    if st.session_state.exp_step == 1 and "condition" not in st.session_state:
        st.session_state.condition = assign_condition(CONDITIONS)
    st.session_state.exp_step += 1

def _back():
    st.session_state.exp_step = max(1, st.session_state.exp_step - 1)

def _build_qualtrics_url():
    qs = urllib.parse.urlencode({
        "pid": st.session_state.pid,
        "cond": st.session_state.condition,
        "lang": st.session_state.language,
    })
    return QUALTRICS_BASE_URL + ("&" if "?" in QUALTRICS_BASE_URL else "?") + qs

def get_connection():
    return psycopg2.connect(st.secrets["db"]["url"], sslmode="require")

def assign_condition(conditions: list[str]) -> str:
    with get_connection() as conn, conn.cursor() as cur:
        for c in conditions:
            cur.execute(
                "insert into condition_counts (condition) values (%s) on conflict do nothing",
                (c,),
            )
        cur.execute(
            "select condition, count from condition_counts where condition = any(%s)",
            (conditions,),
        )
        rows = cur.fetchall()
        min_count = min(n for _, n in rows)
        chosen = random.choice([c for c, n in rows if n == min_count])
        cur.execute(
            "update condition_counts set count = count + 1, updated_at = now() where condition = %s",
            (chosen,),
        )
        return chosen

# =========================================================
# Feature table (LANGUAGE AWARE)
# =========================================================

def _features_to_table(features: dict) -> pd.DataFrame:
    lang = st.session_state.language
    var_labels = get_var_labels(lang)
    var_units = get_var_units(lang)
    var_desc = get_var_descriptions(lang)
    group_labels = get_group_labels(lang)

    rows = []
    grouped_vars = {v for vs in var_groups.values() for v in vs}

    for group, vars_ in var_groups.items():
        group_label = group_labels.get(group, group)

        for v in vars_:
            if v not in features:
                continue

            label = var_labels.get(v, v)
            unit = var_units.get(v)
            if unit:
                label = f"{label} ({unit})"

            value = features[v]
            if isinstance(value, (int, np.integer)) and v in binary_vars:
                value = "Yes" if (lang == "en" and value == 1) else "No" if lang == "en" else "Ja" if value == 1 else "Nee"

            rows.append({
                "Categorie": group_label,
                "Variabele": label,
                "Waarde": value,
                "Beschrijving": var_desc.get(v, ""),
            })

    return pd.DataFrame(rows)

def step_0_language():
    st.header(t("language_title"))
    st.markdown(t("language_text"))

    col1, col2 = st.columns(2)
    with col1:
        if st.button(t("language_nl")):
            st.session_state.language = "nl"
            st.session_state.exp_step = 1
    with col2:
        if st.button(t("language_en")):
            st.session_state.language = "en"
            st.session_state.exp_step = 1

def step_1_consent():
    st.header(t("welcome_title"))
    st.markdown(TEXT[st.session_state.language]["consent_text"])
    consent = st.checkbox(t("consent_checkbox"))
    st.button(t("next"), disabled=not consent, on_click=_next)

def step_2_assignment():
    st.header(t("system_explanation"))
    st.info(f"{t('assigned_version')}: **{st.session_state.condition}**")
    st.button(t("next"), on_click=_next)
    st.button(t("back"), on_click=_back)

def step_3_scenario():
    scenario = _get_scenario()
    st.header(t("scenario_header"))
    st.markdown(scenario.narrative_markdown)
    st.markdown(t("project_features"))

    df = _features_to_table(scenario.features)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if st.button(t("predict")):
        st.session_state.show_results = True

    if st.session_state.get("show_results"):
        st.subheader(t("prediction_header"))
        p = XAI_DIR / "prediction.txt"
        if p.exists():
            st.markdown(html.unescape(p.read_text()))
        st.subheader(f"{t('explanation_header')} ({st.session_state.condition})")

    st.button(t("continue"), on_click=_next)
    st.button(t("back"), on_click=_back)

def step_4_redirect():
    st.header(t("questionnaire"))
    url = _build_qualtrics_url()
    st.link_button(t("open_questionnaire"), url)
    st.caption(t("copy_link"))
    st.code(url)
    st.button(t("back"), on_click=_back)

def run_experiment():
    _init_participant_state()
    _progress()

    step = st.session_state.exp_step
    if step == 0:
        step_0_language()
    elif step == 1:
        step_1_consent()
    elif step == 2:
        step_2_assignment()
    elif step == 3:
        step_3_scenario()
    else:
        step_4_redirect()
