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
from dictionaries import DEFAULT_LANGUAGE, var_groups, binary_vars, get_var_labels, get_var_descriptions, get_group_labels, get_var_units
from experiment_config import CONDITIONS, COND_LABELS_EN, QUALTRICS_BASE_URL, SCENARIOS, xai_path, XAI_DIR

def _init_participant_state():
    if "pid" not in st.session_state: #participant id
        st.session_state.pid = str(uuid.uuid4())
    if "scenario_id" not in st.session_state:
        st.session_state.scenario_id = SCENARIOS[0].scenario_id
    if "exp_step" not in st.session_state:
        st.session_state.exp_step = 0
    if "language" not in st.session_state:
        st.session_state.language = DEFAULT_LANGUAGE

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

#to prevent condition assignment by streamlit health checks (was updating the condition countdatabase)

def _back():
    st.session_state.exp_step = max(1, st.session_state.exp_step - 1)

def _progress():
    total_steps = 5
    st.progress(st.session_state.exp_step / total_steps)

    if st.session_state.language == "en":
        st.caption(f"Step {st.session_state.exp_step} of {total_steps}")
    else:
        st.caption(f"Stap {st.session_state.exp_step} van {total_steps}")

def _build_qualtrics_url():
    params = {
        "pid": st.session_state.pid,
        "cond": st.session_state.condition,
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
                    (c,)
                )

            cur.execute(
                """
                select condition, count
                from condition_counts
                where condition = any(%s)
                """,
                (conditions,)
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
                (chosen,)
            )

            return chosen

def _features_to_table(features: dict) -> pd.DataFrame:
    language = st.session_state.language

    var_labels = get_var_labels(language)
    var_descriptions = get_var_descriptions(language)
    var_units = get_var_units(language)
    group_labels = get_group_labels(language)

    yes_no = ("Yes", "No") if language == "en" else ("Ja", "Nee")

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
                value = yes_no[0] if value == 1 else yes_no[1]

            rows.append({
                "Category": group_label if language == "en" else "Categorie",
                "Variable": label if language == "en" else "Variabele",
                "Value": value if language == "en" else "Waarde",
                "Description": var_descriptions.get(v, "") if language == "en" else "Beschrijving",
            })

    for k, val in features.items():
        if k in grouped_vars:
            continue

        label = var_labels.get(k, k)
        unit = var_units.get(k)
        if unit:
            label = f"{label} ({unit})"

        rows.append({
            "Category": "Other" if language == "en" else "Overig",
            "Variable": label,
            "Value": val,
            "Description": var_descriptions.get(k, ""),
        })

    df = pd.DataFrame(rows)

    #df["Categorie"] = df["Categorie"].where(
    #    df["Categorie"].ne(df["Categorie"].shift()),
    #    ""
    #)
    df.iloc[:, 0] = df.iloc[:, 0].where(df.iloc[:, 0].ne(df.iloc[:, 0].shift()), "")

    return df

def step_0_language():
    st.header("Language selection / Taalkeuze")

    choice = st.radio(
        "Please select your preferred language / Kies uw gewenste taal",
        options=[
            ("English", "en"),
            ("Nederlands", "nl"),
        ],
        format_func=lambda x: x[0],
    )

    st.session_state.language = choice[1]

    if st.session_state.language == "nl":
        st.session_state.scenario_id = "S1_NL"
    else:
        st.session_state.scenario_id = "S1_EN"

    st.button("Continue / Doorgaan", on_click=_next)

def step_1_consent():
    language = st.session_state.language

    if language == "nl":
        st.header("Welkom")
        st.markdown("""
In dit onderzoek maakt u kennis met een prototype van een AI‑gebaseerd 
beslissingsondersteunend systeem voor bouwprojecten. 
U krijgt een projectsituatie te zien, samen met een voorspelling van het risico op 
projectvertraging die door het systeem wordt gegenereerd. Afhankelijk van de versie 
van het systeem die u te zien krijgt, wordt deze voorspelling mogelijk ondersteund 
door aanvullende uitleg. 

Tijdens het experiment wordt u gevraagd om de informatie die het systeem presenteert 
zorgvuldig te bekijken. Het onderzoek richt zich niet op het nemen van beslissingen, 
maar op **uw perceptie van de uitkomsten van het systeem en de bijbehorende uitleg**. 

Het experiment bestaat uit enkele stappen en neemt ongeveer 10 minuten in beslag. 
Na afloop wordt u automatisch doorgestuurd naar een vragenlijst waarin u wordt gevraagd 
uw ervaringen met het systeem te beoordelen. 

Uw deelname is vrijwillig en uw antwoorden worden anoniem verwerkt. U kunt op elk 
moment stoppen met het experiment zonder opgave van reden.
""")
        consent_label = "Ik heb de bovenstaande informatie gelezen en ga akkoord met deelname aan dit onderzoek."
        next_label = "Volgende"

    else:  # ENGLISH
        st.header("Welcome")
        st.markdown("""
In this study, you will be introduced to a prototype of an AI‑based decision‑support system 
for construction projects. You will be presented with a project scenario together with a 
prediction of the risk of project delay generated by the system. Depending on the version 
of the system you are assigned to, this prediction may be supported by additional explanations. 

During the experiment, you are asked to carefully review the information presented by the system. 
The study does not focus on making decisions, but on **your perception of the system’s outputs 
and the accompanying explanations**. 

The experiment consists of several steps and takes approximately 10 minutes to complete. 
After completing the experiment, you will be automatically redirected to a questionnaire in 
which you will be asked to evaluate your experience with the system. 

Participation is voluntary and your responses will be processed anonymously. You may withdraw 
from the experiment at any time without providing a reason.
""")
        consent_label = "I have read the information above and agree to participate in this study."
        next_label = "Next"

    consent = st.checkbox(consent_label)

    col1, col2 = st.columns([1, 1])
    with col2:
        st.button(next_label, disabled=not consent, on_click=_next)

def step_2_assignment():
    language = st.session_state.language
    cond = st.session_state.condition


    if language == "nl":
        st.header("Uitleg van het systeem")
        st.markdown("""
            In de volgende stap krijgt u een projectsituatie te zien, samen met een voorspelling 
            van het risico op projectvertraging die door een AI‑systeem wordt gegenereerd. 
            U bent toegewezen aan een specifieke versie van het systeem. Deze versie verschilt in 
            de manier waarop de voorspelling wordt toegelicht. Hieronder wordt kort uitgelegd hoe 
            de uitleg in uw versie is opgebouwd.
        """)
        assigned_label = "Toegewezen versie"
        next_label = "Volgende"
        back_label = "Terug"

    else:
        st.header("System explanation")
        st.markdown("""
            In the next step, you will be presented with a project scenario together with a prediction 
            of the risk of project delay generated by an AI system. You have been assigned to a specific 
            version of the system. This version differs in the way the prediction is explained. 
            Below, a brief explanation is provided of how the explanation in your version is structured.
        """)
        assigned_label = "Assigned version"
        next_label = "Next"
        back_label = "Back"

    if language == "nl":
        st.info(f"{assigned_label}: **{cond}**")
    else:
        st.info(f"{assigned_label}: **{COND_LABELS_EN[cond]}**")

    if cond == "Black box":
        st.info(
            """
**In deze versie van het systeem wordt alleen de voorspelling getoond.**

Er wordt geen aanvullende uitleg gegeven over hoe het systeem tot deze voorspelling
is gekomen.
"""
        )
 
    if cond == "Black box":
        st.info("""
            **In deze versie van het systeem wordt alleen de voorspelling getoond.**

            Er wordt geen aanvullende uitleg gegeven over hoe het systeem tot deze voorspelling
            is gekomen.
            """
            if language == "nl"
            else
            """
            **In this version of the system, only the prediction is shown.**
            
            No additional explanation is provided about how the system arrived at this prediction.
            """)

    elif cond == "SHAP":
        st.info("""
                **In deze versie van het systeem wordt de voorspelling ondersteund door een visuele uitleg.**

                De uitleg laat zien welke projectkenmerken volgens het systeem het meest 
                bijgedragen aan de voorspelling, en in welke mate deze kenmerken het risico op
                vertraging verhogen of verlagen.
                """
                if language == "nl"
                else
                """
                **In this version of the system, the prediction is supported by a visual explanation.**

                The explanation shows which project characteristics, according to the system, contributed the most to the prediction, 
                and to what extent these characteristics increase or decrease the risk of delay.
                """)

    elif cond == "Regels":
        st.info("""
                **In deze versie van het systeem wordt de voorspelling toegelicht met behulp van regels.**

                Deze regels beschrijven combinaties van projectkenmerken waarvoor de voorspelling
                geldig is. De uitleg geeft inzicht in welke voorwaarden doorslaggevend zijn geweest
                voor de uitkomst.
                """
                if language == "nl"
                else
                """**In this version of the system, the prediction is explained using rules.**

                These rules describe combinations of project characteristics for which the prediction is valid. 
                The explanation provides insight into which conditions were decisive for the outcome.
                """)

    elif cond == "Tegenfeitelijk":
        st.info("""
                **In deze versie van het systeem wordt de voorspelling toegelicht met alternatieve scenario’s.**

                De uitleg laat zien hoe aanpassingen in specifieke projectkenmerken zouden
                kunnen leiden tot een andere voorspelling, bijvoorbeeld een lager risico op
                vertraging.
                """
                if language == "nl"
                else
                """
                **In this version of the system, the prediction is explained with alternative scenarios.**

                The explanation shows how adjustments in specific project characteristics could lead to a different prediction, for example, a lower risk of delay.
                """)

    elif cond == "Surrogaatmodel (beslisboom)":
        st.info("""
                **In deze versie van het systeem wordt de voorspelling toegelicht met een vereenvoudigd model.**

                Dit model geeft een overzicht van de belangrijkste beslisregels die het AI‑systeem
                gebruikt om tot een voorspelling te komen, in een vorm die makkelijker te interpreteren is.
                """
                if language == "nl"
                else
                """
                **In this version of the system, the prediction is explained with a simplified model.**
                
                This model provides an overview of the main decision rules that the AI system uses to arrive at a prediction, in a form that is easier to interpret.
                """)

    st.markdown("""
                In de volgende stap ziet u de projectsituatie en kunt u de voorspelling en bijbehorende uitleg bekijken.
                """
                if language == "nl"
                else
                """
                In the next step, you will see the project situation and can view the prediction and corresponding explanation.
                """)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button(back_label, on_click=_back)
    with col2:
        st.button(next_label, on_click=_next)

def step_3_scenario_nl():
    st.session_state.language = "nl"
    scenario = _get_scenario()

    st.header("Scenario en voorspelling")
    st.markdown(scenario.narrative_markdown)

    if scenario.image_path:
        try:
            st.image(scenario.image_path, use_container_width=True)
        except Exception:
            st.warning("De scenario-afbeelding kon niet worden geladen.")

    st.markdown("#### Projectkenmerken")
    df = _features_to_table(scenario.features)
    st.dataframe(df, use_container_width=True, hide_index=True, 
        column_config={
                "Categorie": st.column_config.TextColumn(
                    "Categorie", width=240),
                "Variabele": st.column_config.TextColumn(
                    "Variabele", width=460),
                "Waarde": st.column_config.TextColumn(
                    "Waarde", width=100),
                "Beschrijving": st.column_config.TextColumn(
                    "Beschrijving", width=950)
                ,}
        )

    st.markdown("---")
    st.markdown("""
        Hieronder kunt u op **Voorspellen** klikken om de AI‑uitkomst en de bijbehorende uitleg te bekijken.

        De invoerwaarden zijn vastgezet voor dit onderzoek en kunnen niet worden aangepast.
    """)

    if "show_results" not in st.session_state:
        st.session_state.show_results = False

    if st.button("Voorspellen", key="step3_predict",):
        st.session_state.show_results = True
    st.markdown("---")
    
    if st.session_state.show_results:
        st.subheader("AI‑uitkomst")

        prediction_path = XAI_DIR / "prediction.txt"

        if prediction_path.exists():
            text = html.unescape(prediction_path.read_text(encoding="utf-8"))

            for line in text.splitlines():
                line = line.strip()
                if not line:
                    st.markdown("")
                    continue

                if ":" in line:
                    left, right = line.split(":", 1)
                    st.markdown(f"{left}: **{right.strip()}**")
                else:
                    st.markdown(f"**{line}**")
            st.markdown(
                "*Deze voorspelling geeft de **verwachte impact van het risico op projectvertraging** weer. "
                "Het AI‑model combineert de kans op vertraging met de verwachte ernst ervan ten opzichte van de geplande projectduur.*"
                )
        else:
            st.warning(f"Voorspellingsbestand niet gevonden: {prediction_path}")

        st.markdown("---")
        
        cond = st.session_state.condition
        st.subheader(f"Uitleg ({cond})")

        if cond == "SHAP":            
            st.markdown(
                """
            <div style="background-color:#fff8f4; border:2px solid #ddd; padding:16px; margin-bottom:16px;">
              <div style="font-weight:600; font-size:17px; margin-bottom:8px;">
                Hoe moet ik deze uitleg interpreteren?
              </div>
              <div style="font-size:15px; line-height:1.5;">
                Deze grafiek laat zien <strong>welke projectkenmerken volgens het model het meest bijdragen</strong>
                aan de voorspelling voor <strong>dit specifieke project</strong>.
                <br><br>
                &#8226; Balken die omhoog wijzen vergroten het voorspelde vertragingsrisico.<br>
                &#8226; Balken die omlaag wijzen verkleinen het voorspelde vertragingsrisico.<br>
                &#8226; Hoe langer de balk, hoe groter de invloed van dat kenmerk volgens het model.
                <br><br>
                Deze uitleg helpt om te begrijpen <strong>waarom het model tot deze voorspelling komt</strong>,
                maar zegt niets over welke kenmerken in alle projecten belangrijk zijn.
              </div>
            </div>
            
            <div style="background-color:#fff8f4; border:1px solid #ddd; padding:16px;">
              <div style="font-weight:600; font-size:17px; margin-bottom:8px;">
                Hoe genereert het systeem deze uitleg?
              </div>
              <div style="font-size:15px; line-height:1.5;">
                Deze uitleg is gebaseerd op een methode die per project bekijkt
                hoeveel ieder projectkenmerk bijdraagt aan de uiteindelijke voorspelling.
                <br><br>
                Het model vergelijkt daarbij de voorspelling met en zonder specifieke kenmerken.
                Op basis van dit verschil wordt bepaald of een kenmerk de voorspelling verhoogt of verlaagt.
                <br><br>
                De uitleg is specifiek voor dit project en deze voorspelling.
              </div>
            </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

        elif cond == "Regels":
            st.markdown(
                """
        <div style="background-color:#fff8f4; border:2px solid #ddd; padding:16px; margin-bottom:16px;">
          <div style="font-weight:600; font-size:17px; margin-bottom:8px;">
            Hoe moet ik deze uitleg interpreteren?
          </div>
          <div style="font-size:15px; line-height:1.5;">
            Deze regel laat zien <strong>onder welke combinatie van projectkenmerken</strong>
            het AI-systeem tot een vergelijkbare voorspelling komt als bij dit project.
            <br><br>
            Als aan de genoemde voorwaarden wordt voldaan,
            hoort daar volgens het model deze inschatting van de verwachte vertraging bij.
            <br><br>
            Zie deze regel als:
            <em>de belangrijkste kenmerken waarop het model zich in dit geval baseert.</em>
          </div>
        </div>
        
        <div style="background-color:#fff8f4; border:1px solid #ddd; padding:16px;">
          <div style="font-weight:600; font-size:17px; margin-bottom:8px;">
            Hoe genereert het systeem deze uitleg?
          </div>
          <div style="font-size:15px; line-height:1.5;">
            Deze uitleg is afgeleid van het oorspronkelijke AI-model met behulp van
            een eenvoudiger model dat beter uitlegbaar is.
            <br><br>
            Dit vereenvoudigde model probeert het gedrag van het AI-model
            zo goed mogelijk te benaderen in de buurt van deze voorspelling.
            <br><br>
            De regel is daardoor een benadering en geen exacte weergave
            van alle interne berekeningen van het oorspronkelijke model.
            </div>
          </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)


        elif cond == "Tegenfeitelijk":
            st.markdown(
                """
        <div style="background-color:#fff8f4; border:2px solid #ddd; padding:16px; margin-bottom:16px;">
          <div style="font-weight:600; font-size:17px; margin-bottom:8px;">
            Hoe moet ik deze uitleg interpreteren?
          </div>
          <div style="font-size:15px; line-height:1.5;">
            Deze tabel vergelijkt het oorspronkelijke project met een alternatief scenario
            dat volgens het model leidt tot een <strong>lager vertragingsrisico</strong>.
            <br><br>
            &#8226; <strong>Origineel</strong>: de waarde uit het projectscenario dat u zojuist heeft gezien (de invoer voor dit project).<br>
            &#8226; <strong>Tegenfeitelijk</strong>: een alternatief scenario met aangepaste waarden dat volgens het model gunstiger uitpakt.<br>
            &#8226; <strong>Verschil</strong>: het verschil tussen Tegenfeitelijk en Origineel.
            <br><br>
            In de cellen staan waarden inclusief eenheden. De rijen geven aan
            <strong>welke projectkenmerken veranderd zouden moeten worden</strong> om volgens het model een lagere voorspelde vertraging te krijgen.
            <br><br>
            Dit is een modelmatige “wat-als” uitleg. Het betekent niet automatisch dat veranderingen haalbaar,
            wenselijk of volledig onder controle zijn in de praktijk.
          </div>
        </div>
        
        <div style="background-color:#fff8f4; border:1px solid #ddd; padding:16px;">
          <div style="font-weight:600; font-size:17px; margin-bottom:8px;">
            Hoe genereert het systeem deze uitleg?
          </div>
          <div style="font-size:15px; line-height:1.5;">
            Het systeem zoekt naar een alternatief scenario dat zo veel mogelijk lijkt op het huidige project,
            maar waarvoor de voorspelling duidelijk gunstiger is.
            <br><br>
            Daarbij probeert het systeem <strong>zo weinig mogelijk kenmerken</strong> te veranderen en
            vooral kenmerken aan te passen die volgens het model invloed hebben op de uitkomst.
            <br><br>
            Het resultaat is een voorbeeld van een “minimale” aanpassing die volgens het model
            voldoende is om de voorspelling te verbeteren.
          </div>
        </div>
                """,
                unsafe_allow_html=True,
            )
        
            st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

        elif cond == "Surrogaatmodel (beslisboom)":
            st.markdown(
                """
        <div style="background-color:#fff8f4; border:2px solid #ddd; padding:16px; margin-bottom:16px;">
          <div style="font-weight:600; font-size:17px; margin-bottom:8px;">
            Hoe moet ik deze uitleg interpreteren?
          </div>
          <div style="font-size:15px; line-height:1.5;">
            Deze beslisboom geeft een <strong>vereenvoudigd overzicht</strong>
            van hoe het AI-systeem in grote lijnen tot een voorspelling komt.
            <br><br>
            De eerste splitsingen laten zien
            welke projectkenmerken vaak als belangrijk worden gezien.
            <br><br>
            Zie dit als:
            <em>een globale indruk van hoe het model redeneert</em>,
            niet als een exacte beschrijving van elke afzonderlijke voorspelling.
          </div>
        </div>
        
        <div style="background-color:#fff8f4; border:1px solid #ddd; padding:16px;">
          <div style="font-weight:600; font-size:17px; margin-bottom:8px;">
            Hoe genereert het systeem deze uitleg?
          </div>
          <div style="font-size:15px; line-height:1.5;">
            Deze beslisboom is getraind om het gedrag van het oorspronkelijke AI-model
            zo goed mogelijk na te bootsen met een eenvoudiger en beter uitlegbaar model.
            <br><br>
            Door deze vereenvoudiging wordt de uitleg overzichtelijker,
            maar gaat ook een deel van de nauwkeurigheid verloren.
            <br><br>
            De boom geeft daarom inzicht in algemene patronen,
            niet in alle details van het onderliggende model.
          </div>
        </div>
                """,
                unsafe_allow_html=True,
            )
        
            st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

        if cond == "Black box":
            st.info("In deze versie van het systeem wordt geen uitleg bij de voorspelling gegeven.")

        elif cond == "SHAP":
            p = xai_path(cond)
            if p.exists():
                col_spacer_left, col_img, col_spacer_right = st.columns([1, 8, 1])
                with col_img:
                    st.image(str(p), use_container_width=True)
            else:
                st.warning(f"Afbeelding niet gevonden: {p}")
    
        elif cond == "Surrogaatmodel (beslisboom)":
            p = xai_path(cond)
            if p.exists():
                st.image(str(p), use_container_width=True)  # maximaal groot
            else:
                st.warning(f"Afbeelding niet gevonden: {p}")

        elif cond == "Regels":
            p = xai_path(cond)
            if p.exists():
                text = html.unescape(p.read_text(encoding="utf-8"))

                lines = [l.strip() for l in text.splitlines() if l.strip()]

                st.markdown("**Deze voorspelling wordt toegelicht met behulp van de volgende beslisregels:**")
                st.markdown(
                    "\n".join(f"- {line}" for line in lines[1:])
                )
            else:
                st.warning(f"Regels niet gevonden: {p}")

        elif cond == "Tegenfeitelijk":
            p = xai_path(cond)
            if p.exists():
                df = pd.read_csv(p, sep=";")

                for col in ["Origineel", "Tegenfeitelijk", "Verschil"]:
                    df[col] = df[col].astype(str)

                last_idx = df.index[-1]

                def bold_last_row(row):
                    if row.name == last_idx:
                        return ["font-weight: 600;"] * len(row)
                    return [""] * len(row)
        
                styler = df.style.apply(bold_last_row, axis=1)

                st.dataframe(
                    styler,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning(f"Tabel niet gevonden: {p}")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("Terug", key="step3_back", on_click=_back)
    with col2:
        st.button(
            "Doorgaan naar vragenlijst", key="step3_next",
            disabled=not st.session_state.get("show_results", False),
            on_click=_next
        )

def step_3_scenario_en():
    scenario = _get_scenario()

    st.header("Scenario and prediction")
    st.markdown(scenario.narrative_markdown)

    if scenario.image_path:
        try:
            st.image(scenario.image_path, use_container_width=True)
        except Exception:
            st.warning("The scenario image could not be loaded.")

    st.markdown("#### Project features")

    df = _features_to_table(scenario.features)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Category": st.column_config.TextColumn("Category", width=240),
            "Variable": st.column_config.TextColumn("Variable", width=460),
            "Value": st.column_config.TextColumn("Value", width=100),
            "Description": st.column_config.TextColumn("Description", width=950),
        },
    )

    st.markdown("---")
    st.markdown(
        """
Below you can click **Predict** to view the AI output and the accompanying explanation.
The input values are fixed for this study and cannot be adjusted.
"""
    )

    if "show_results" not in st.session_state:
        st.session_state.show_results = False

    if st.button("Predict", key="step3_predict",):
        st.session_state.show_results = True
    st.markdown("---")
    
    if st.session_state.show_results:
        st.subheader("AI‑output")

        prediction_path = XAI_DIR / "prediction_en.txt"

        if prediction_path.exists():
            text = html.unescape(prediction_path.read_text(encoding="utf-8"))

            for line in text.splitlines():
                line = line.strip()
                if not line:
                    st.markdown("")
                    continue

                if ":" in line:
                    left, right = line.split(":", 1)
                    st.markdown(f"{left}: **{right.strip()}**")
                else:
                    st.markdown(f"**{line}**")
            st.markdown(
                    "*This prediction represents the **expected impact of the risk of project delay**. "
                    "The AI model combines the probability of delay with its expected severity relative to "
                    "the planned project duration.*"
                )
        else:
            st.warning(f"Prediction file not found: {prediction_path}")

        st.markdown("---")
        
        cond = st.session_state.condition
        st.subheader(f"Explanation ({COND_LABELS_EN[cond]})")

        if cond == "SHAP":            
            st.markdown(
                """
                    <div style="background-color:#fff8f4; border:2px solid #ddd; padding:16px; margin-bottom:16px;">
                    <div style="font-weight:600; font-size:17px; margin-bottom:8px;">
                    How should I interpret this explanation?
                    </div>
                    <div style="font-size:15px; line-height:1.5;">
                    This chart shows <strong>which project characteristics contribute the most according to the model</strong>
                    to the prediction for <strong>this specific project</strong>.
                    <br><br>
                    &#8226; Bars pointing upward increase the predicted delay risk.<br>
                    &#8226; Bars pointing downward decrease the predicted delay risk.<br>
                    &#8226; The longer the bar, the greater the influence of that characteristic according to the model.
                    <br><br>
                    This explanation helps to understand <strong>why the model arrives at this prediction</strong>,
                    but does not indicate which characteristics are important across all projects.
                    </div>
                    </div>
                    <div style="background-color:#fff8f4; border:1px solid #ddd; padding:16px;">
                    <div style="font-weight:600; font-size:17px; margin-bottom:8px;">
                    How does the system generate this explanation?
                    </div>
                    <div style="font-size:15px; line-height:1.5;">
                    This explanation is based on a method that examines, per project,
                    how much each project characteristic contributes to the final prediction.
                    <br><br>
                    The model compares the prediction with and without specific characteristics.
                    Based on this difference, it determines whether a characteristic increases or decreases the prediction.
                    <br><br>
                    The explanation is specific to this project and this prediction.
                    </div>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

        elif cond == "Regels":
            st.markdown(
                """
                    <div style="background-color:#fff8f4; border:2px solid #ddd; padding:16px; margin-bottom:16px;">
                    <div style="font-weight:600; font-size:17px; margin-bottom:8px;">
                    How should I interpret this explanation?
                    </div>
                    <div style="font-size:15px; line-height:1.5;">
                    This rule shows <strong>under which combination of project characteristics</strong>
                    the AI system arrives at a prediction similar to that of this project.
                    <br><br>
                    If the stated conditions are met,
                    the model associates them with this estimated level of expected delay.
                    <br><br>
                    See this rule as:
                    <em>the key characteristics on which the model bases its reasoning in this case.</em>
                    </div>
                    </div>
                    <div style="background-color:#fff8f4; border:1px solid #ddd; padding:16px;">
                    <div style="font-weight:600; font-size:17px; margin-bottom:8px;">
                    How does the system generate this explanation?
                    </div>
                    <div style="font-size:15px; line-height:1.5;">
                    This explanation is derived from the original AI model using
                    a simpler model that is easier to interpret.
                    <br><br>
                    This simplified model attempts to approximate the behavior of the AI model
                    as closely as possible in the vicinity of this prediction.
                    <br><br>
                    As a result, the rule is an approximation and not an exact representation
                    of all internal calculations of the original model.
                    </div>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)


        elif cond == "Tegenfeitelijk":
            st.markdown(
                """
                    <div style="background-color:#fff8f4; border:2px solid #ddd; padding:16px; margin-bottom:16px;">
                    <div style="font-weight:600; font-size:17px; margin-bottom:8px;">
                    How should I interpret this explanation?
                    </div>
                    <div style="font-size:15px; line-height:1.5;">
                    This table compares the original project with an alternative scenario
                    that leads to a <strong>lower predicted delay risk</strong> according to the model.
                    <br><br>
                    &#8226; <strong>Original</strong>: the value from the project scenario you just saw.<br>
                    &#8226; <strong>Counterfactual</strong>: an alternative scenario with adjusted values.<br>
                    &#8226; <strong>Difference</strong>: the difference between Counterfactual and Original.
                    <br><br>
                    This is a model‑based what‑if explanation and does not guarantee that such changes
                    are feasible or desirable in practice.
                    </div>
                    </div>
                    <div style="background-color:#fff8f4; border:1px solid #ddd; padding:16px;">
                    <div style="font-weight:600; font-size:17px; margin-bottom:8px;">
                    How does the system generate this explanation?
                    </div>
                    <div style="font-size:15px; line-height:1.5;">
                    The system searches for an alternative scenario that closely resembles
                    the current project but results in a more favorable prediction.
                    <br><br>
                    The goal is to change as few characteristics as possible, focusing on those
                    that the model considers influential.
                    </div>
                    </div>
                """,
                unsafe_allow_html=True,
            )
        
            st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

        elif cond == "Surrogaatmodel (beslisboom)":
            st.markdown(
                """
                    <div style="background-color:#fff8f4; border:2px solid #ddd; padding:16px; margin-bottom:16px;">
                    <div style="font-weight:600; font-size:17px; margin-bottom:8px;">
                    How should I interpret this explanation?
                    </div>
                    <div style="font-size:15px; line-height:1.5;">
                    This decision tree provides a <strong>simplified overview</strong>
                    of how the AI system generally arrives at a prediction.
                    <br><br>
                    See this as:
                    <em>a high‑level impression of the model’s reasoning</em>,
                    not an exact description of every individual prediction.
                    </div>
                    </div>
                    <div style="background-color:#fff8f4; border:1px solid #ddd; padding:16px;">
                    <div style="font-weight:600; font-size:17px; margin-bottom:8px;">
                    How does the system generate this explanation?
                    </div>
                    <div style="font-size:15px; line-height:1.5;">
                    This decision tree approximates the behavior of the original AI model
                    using a simpler and more interpretable structure.
                    <br><br>
                    This simplification improves readability
                    but omits some detail and precision.
                    </div>
                    </div>
                """,
                unsafe_allow_html=True,
            )
        
            st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

        if cond == "Black box":
            st.info("In this version of the system, no explanation is provided for the prediction.")

        elif cond == "SHAP":
            p = xai_path(cond)
            if p.exists():
                col_spacer_left, col_img, col_spacer_right = st.columns([1, 8, 1])
                with col_img:
                    st.image(str(p), use_container_width=True)
            else:
                st.warning(f"Image not found: {p}")
    
        elif cond == "Surrogaatmodel (beslisboom)":
            p = xai_path(cond)
            if p.exists():
                st.image(str(p), use_container_width=True)  
            else:
                st.warning(f"Image not found: {p}")

        elif cond == "Regels":
            p = xai_path(cond)
            if p.exists():
                text = html.unescape(p.read_text(encoding="utf-8"))

                lines = [l.strip() for l in text.splitlines() if l.strip()]

                st.markdown("**These predictions are explained using the following decision rules:**")
                st.markdown(
                    "\n".join(f"- {line}" for line in lines[1:])
                )
            else:
                st.warning(f"Rules not found: {p}")

        elif cond == "Tegenfeitelijk":
            p = xai_path(cond)
            if p.exists():
                df = pd.read_csv(p, sep=";")

                for col in ["Original", "Counterfactual", "Difference"]:
                    df[col] = df[col].astype(str)

                last_idx = df.index[-1]

                def bold_last_row(row):
                    if row.name == last_idx:
                        return ["font-weight: 600;"] * len(row)
                    return [""] * len(row)
        
                styler = df.style.apply(bold_last_row, axis=1)

                st.dataframe(
                    styler,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning(f"Table not found: {p}")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("Back", key="step3_back", on_click=_back)
    with col2:
        st.button(
            "Continue to questionnaire", key="step3_next",
            disabled=not st.session_state.get("show_results", False),
            on_click=_next
        )

def step_3_scenario():
    if st.session_state.language == "nl":
        step_3_scenario_nl()
    else:
        step_3_scenario_en()

def step_4_redirect():
    language = st.session_state.language

    if language == "nl":
        st.header("Vragenlijst")
        intro = "Klik op de knop hieronder om door te gaan naar de vragenlijst."
        button = "Open vragenlijst"
        back_label = "Terug"
        caption = "Werkt de knop niet? Kopieer dan de onderstaande link en plak deze in uw browser."
    else:
        st.header("Questionnaire")
        intro = "Click the button below to continue to the questionnaire."
        button = "Open questionnaire"
        back_label = "Back"
        caption = "If the button does not work, copy the link below and paste it into your browser."

    st.markdown(intro)
    url = _build_qualtrics_url()
    st.link_button(button, url)
    st.caption(caption)
    st.code(url, language="text")

    st.markdown("---")
    st.button(back_label, on_click=_back)

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
