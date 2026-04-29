from __future__ import annotations

import uuid
import urllib.parse
from dataclasses import asdict
import html
import pandas as pd
import streamlit as st
import numpy as np
import json
from pathlib import Path
import random
import psycopg2

from dictionaries import (
    var_groups,
    binary_vars,
    get_var_labels,
    get_var_descriptions,
    get_var_units,
    get_group_labels,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
)

from experiment_config import (
    CONDITIONS,
    QUALTRICS_BASE_URL,
    SCENARIOS,
    xai_path,
    XAI_DIR,
)


def _init_participant_state():
    if "pid" not in st.session_state:
        st.session_state.pid = str(uuid.uuid4())
    if "scenario_id" not in st.session_state:
        st.session_state.scenario_id = SCENARIOS[0].scenario_id
    if "exp_step" not in st.session_state:
        st.session_state.exp_step = 0
    if "language" not in st.session_state:
        st.session_state.language = DEFAULT_LANGUAGE


def _get_text(dutch: str, english: str) -> str:
    return english if st.session_state.language == "en" else dutch


def _get_scenario():
    sid = st.session_state.scenario_id
    for s in SCENARIOS:
        if s.scenario_id == sid:
            return s
    return SCENARIOS[0]


def assign_condition_if_needed():
    if "condition" not in st.session_state:
        st.session_state.condition = assign_condition(CONDITIONS)


def _next():
    if st.session_state.exp_step == 1:
        assign_condition_if_needed()
    st.session_state.exp_step += 1


def _back():
    st.session_state.exp_step = max(0, st.session_state.exp_step - 1)


def _progress():
    st.progress(st.session_state.exp_step / 4)
    st.caption(
        _get_text(
            f"Stap {st.session_state.exp_step} van 4",
            f"Step {st.session_state.exp_step} of 4",
        )
    )


def _build_qualtrics_url():
    params = {
        "pid": st.session_state.pid,
        "cond": st.session_state.condition,
        "lang": st.session_state.language,
    }
    qs = urllib.parse.urlencode(params)
    if "?" in QUALTRICS_BASE_URL:
        return QUALTRICS_BASE_URL + "&" + qs
    return QUALTRICS_BASE_URL + "?" + qs


def get_connection():
    return psycopg2.connect(st.secrets["db"]["url"], sslmode="require")


def assign_condition(conditions: list[str]) -> str:
    with get_connection() as conn:
        with conn.cursor() as cur:
            for c in conditions:
                cur.execute(
                    """
                    insert into condition_counts (condition)
                    values (%s)
                    on conflict (condition) do nothing
                    """,
                    (c,),
                )
            cur.execute(
                """
                select condition, count
                from condition_counts
                where condition = any(%s)
                """,
                (conditions,),
            )
            rows = cur.fetchall()
            min_count = min(n for _, n in rows)
            candidates = [c for c, n in rows if n == min_count]
            chosen = random.choice(candidates)
            cur.execute(
                """
                update condition_counts
                set count = count + 1,
                    updated_at = now()
                where condition = %s
                """,
                (chosen,),
            )
            return chosen


def _features_to_table(features: dict) -> pd.DataFrame:
    language = st.session_state.language
    var_labels = get_var_labels(language)
    var_descriptions = get_var_descriptions(language)
    var_units = get_var_units(language)
    group_labels = get_group_labels(language)

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
            waarde = features[v]
            if isinstance(waarde, (int, np.integer)) and v in binary_vars:
                waarde = (
                    _get_text("Ja", "Yes") if waarde == 1 else _get_text("Nee", "No")
                )
            rows.append(
                {
                    "Categorie": group_label,
                    "Variabele": label,
                    "Waarde": waarde,
                    "Beschrijving": var_descriptions.get(v, ""),
                }
            )

    for k, val in features.items():
        if k in grouped_vars:
            continue
        label = var_labels.get(k, k)
        unit = var_units.get(k)
        if unit:
            label = f"{label} ({unit})"
        rows.append(
            {
                "Categorie": _get_text("Overig", "Other"),
                "Variabele": label,
                "Waarde": val,
                "Beschrijving": var_descriptions.get(k, ""),
            }
        )

    df = pd.DataFrame(rows)
    df["Categorie"] = df["Categorie"].where(
        df["Categorie"].ne(df["Categorie"].shift()), ""
    )
    return df


def step_0_language():
    st.header("Language / Taal")
    st.radio(
        "Select your preferred language",
        options=SUPPORTED_LANGUAGES,
        format_func=lambda x: "English" if x == "en" else "Nederlands",
        key="language",
    )
    st.button("Continue", on_click=_next)


def step_1_consent():
    st.header(_get_text("Welkom", "Welcome"))

    st.markdown(
        _get_text(
            """
In dit onderzoek maakt u kennis met een prototype van een AI‑gebaseerd
beslissingsondersteunend systeem voor bouwprojecten.
U krijgt een projectsituatie te zien, samen met een voorspelling van het risico op
projectvertraging die door het systeem wordt gegenereerd.
Afhankelijk van de versie van het systeem die u te zien krijgt, wordt deze voorspelling
mogelijk ondersteund door aanvullende uitleg.
Tijdens het experiment wordt u gevraagd om de informatie zorgvuldig te bekijken.
Het onderzoek richt zich niet op het nemen van beslissingen, maar op uw perceptie van
de uitkomsten van het systeem en de bijbehorende uitleg.
Het experiment duurt ongeveer 10 minuten.
""",
            """
In this study, you will interact with a prototype AI-based decision-support system
for construction projects.
You will be presented with a project scenario, together with a prediction of project
delay risk generated by the system.
Depending on the system version you are assigned to, this prediction may be supported
by additional explanations.
During the experiment, you are asked to carefully review the information provided.
The study focuses on your perception of the system’s outputs and explanations,
not on decision-making.
The experiment takes approximately 10 minutes.
""",
        )
    )

    consent = st.checkbox(
        _get_text(
            "Ik ga akkoord met deelname aan dit onderzoek.",
            "I consent to participate in this study.",
        )
    )
    st.button(
        _get_text("Volgende", "Next"),
        disabled=not consent,
        on_click=_next,
    )

def step_2_assignment():
    st.header(_get_text("Uitleg van het systeem", "System explanation"))

    st.markdown(
        _get_text(
            """
In de volgende stap krijgt u een projectsituatie te zien, samen met een voorspelling
van het risico op projectvertraging die door een AI‑systeem wordt gegenereerd.

U bent toegewezen aan een specifieke versie van het systeem. Deze versie verschilt in
de manier waarop de voorspelling wordt toegelicht. Hieronder wordt kort uitgelegd hoe
de uitleg in uw versie is opgebouwd.
""",
            """
In the next step, you will see a project scenario together with a prediction of
project delay risk generated by an AI system.

You have been assigned to a specific version of the system. This version differs in
the way the prediction is explained. Below, you will find a brief explanation of how
the explanation in your version is structured.
""",
        )
    )

    cond = st.session_state.condition
    st.info(
        _get_text(
            f"Toegewezen versie: **{cond}**",
            f"Assigned version: **{cond}**",
        )
    )

    if cond == "Black box":
        st.info(
            _get_text(
                """
**In deze versie van het systeem wordt alleen de voorspelling getoond.**

Er wordt geen aanvullende uitleg gegeven over hoe het systeem tot deze voorspelling
is gekomen.
""",
                """
**In this version of the system, only the prediction is shown.**

No additional explanation is provided about how the system arrived at this prediction.
""",
            )
        )

    elif cond == "SHAP":
        st.info(
            _get_text(
                """
**In deze versie van het systeem wordt de voorspelling ondersteund door een visuele uitleg.**

De uitleg laat zien welke projectkenmerken volgens het systeem het meest 
bijgedragen aan de voorspelling, en in welke mate deze kenmerken het risico op
vertraging verhogen of verlagen.
""",
                """
**In this version of the system, the prediction is supported by a visual explanation.**

The explanation shows which project characteristics contributed most to the prediction,
and to what extent these characteristics increase or decrease the delay risk.
""",
            )
        )

    elif cond == "Regels":
        st.info(
            _get_text(
                """
**In deze versie van het systeem wordt de voorspelling toegelicht met behulp van regels.**

Deze regels beschrijven combinaties van projectkenmerken waarvoor de voorspelling
geldig is. De uitleg geeft inzicht in welke voorwaarden doorslaggevend zijn geweest
voor de uitkomst.
""",
                """
**In this version of the system, the prediction is explained using rules.**

These rules describe combinations of project characteristics for which the prediction
applies and provide insight into which conditions were decisive for the outcome.
""",
            )
        )

    elif cond == "Tegenfeitelijk":
        st.info(
            _get_text(
                """
**In deze versie van het systeem wordt de voorspelling toegelicht met alternatieve scenario’s.**

De uitleg laat zien hoe aanpassingen in specifieke projectkenmerken zouden
kunnen leiden tot een andere voorspelling, bijvoorbeeld een lager risico op
vertraging.
""",
                """
**In this version of the system, the prediction is explained using alternative scenarios.**

The explanation shows how changes in specific project characteristics could lead to a
different prediction, such as a lower delay risk.
""",
            )
        )

    elif cond == "Surrogaatmodel (beslisboom)":
        st.info(
            _get_text(
                """
**In deze versie van het systeem wordt de voorspelling toegelicht met een vereenvoudigd model.**

Dit model geeft een overzicht van de belangrijkste beslisregels die het AI‑systeem
gebruikt om tot een voorspelling te komen, in een vorm die makkelijker te interpreteren is.
""",
                """
**In this version of the system, the prediction is explained using a simplified model.**

This model provides an overview of the most important decision rules used by the AI
system to arrive at a prediction, in a form that is easier to interpret.
""",
            )
        )

    st.markdown(
        _get_text(
            "In de volgende stap ziet u de projectsituatie en kunt u de voorspelling en bijbehorende uitleg bekijken.",
            "In the next step, you will see the project scenario and can review the prediction and explanation.",
        )
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button(_get_text("Terug", "Back"), key="step2_back", on_click=_back)
    with col2:
        st.button(_get_text("Volgende", "Next"), key="step2_next", on_click=_next)


def step_3_scenario():
    scenario = _get_scenario()
    st.header(_get_text("Scenario en voorspelling", "Scenario and prediction"))
    st.markdown(scenario.narrative_markdown)

    if scenario.image_path:
        try:
            st.image(scenario.image_path, use_container_width=True)
        except Exception:
            st.warning(
                _get_text(
                    "De scenario-afbeelding kon niet worden geladen.",
                    "The scenario image could not be loaded.",
                )
            )

    st.markdown(_get_text("#### Projectkenmerken", "#### Project characteristics"))
    df = _features_to_table(scenario.features)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        _get_text(
            """Hieronder kunt u op **Voorspellen** klikken om de AI‑uitkomst en de bijbehorende uitleg te bekijken.

De invoerwaarden zijn vastgezet voor dit onderzoek en kunnen niet worden aangepast.""",
            """Click **Predict** below to view the AI output and the accompanying explanation.

The input values are fixed for this study and cannot be modified.""",
        )
    )

    if "show_results" not in st.session_state:
        st.session_state.show_results = False

    if st.button(_get_text("Voorspellen", "Predict"), key="step3_predict"):
        st.session_state.show_results = True

    st.markdown("---")

    if st.session_state.show_results:
        st.subheader(_get_text("AI‑uitkomst", "AI output"))

        prediction_path = XAI_DIR / "prediction.txt"
        if prediction_path.exists():
            text = html.unescape(prediction_path.read_text(encoding="utf-8"))
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    st.markdown("")
                elif ":" in line:
                    left, right = line.split(":", 1)
                    st.markdown(f"{left}: **{right.strip()}**")
                else:
                    st.markdown(f"**{line}**")
        else:
            st.warning(
                _get_text(
                    f"Voorspellingsbestand niet gevonden: {prediction_path}",
                    f"Prediction file not found: {prediction_path}",
                )
            )

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button(_get_text("Terug", "Back"), key="step3_back", on_click=_back)
    with col2:
        st.button(
            _get_text("Doorgaan naar vragenlijst", "Continue to questionnaire"),
            key="step3_next",
            disabled=not st.session_state.get("show_results", False),
            on_click=_next,
        )


def step_4_redirect():
    st.header(_get_text("Vragenlijst", "Questionnaire"))
    st.markdown(
        _get_text(
            "Klik op de knop hieronder om door te gaan naar de vragenlijst.",
            "Click the button below to proceed to the questionnaire.",
        )
    )

    url = _build_qualtrics_url()
    st.link_button(_get_text("Open vragenlijst", "Open questionnaire"), url)
    st.caption(
        _get_text(
            "Werkt de knop niet? Kopieer dan de onderstaande link en plak deze in uw browser.",
            "If the button does not work, copy the link below and paste it into your browser.",
        )
    )
    st.code(url, language="text")

    st.markdown("---")
    st.button(_get_text("Terug", "Back"), key="step4_back", on_click=_back)

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
