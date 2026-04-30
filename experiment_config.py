from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from dictionaries import EXPERIMENT_SCENARIO_PRESET

EXPERIMENT_MODE: bool = True

CONDITIONS = [
    "Black box",
    "SHAP",
    "Regels",
    "Tegenfeitelijk",
    "Surrogaatmodel (beslisboom)",
]

COND_LABELS_EN = {
    "Black box": "Black-box model",
    "SHAP": "Feature importance (SHAP)",
    "Regels": "Rule-based explanation",
    "Tegenfeitelijk": "Counterfactual explanation",
    "Surrogaatmodel (beslisboom)": "Surrogate model (decision tree)",
}

QUALTRICS_BASE_URL = "https://qualtricsxmp3nt7g5jg.qualtrics.com/jfe/form/SV_862H83gyRZHYjoa"

@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    narrative_markdown: str
    image_path: str | None
    features: dict

SCENARIOS = [
    Scenario(
        scenario_id="S1",
        title="Scenario 1",
        narrative_markdown=(
            """### Project scenario\n"""
            "U bent betrokken als projectmanager bij een middelgroot utiliteitsproject dat zich bevindt in de voorbereidingsfase, kort vóór de start van de uitvoeringsfase.\n\n"
            "Het project betreft de nieuwbouw van een multifunctioneel gebouw in een stedelijke omgeving. De opdrachtgever is een professionele partij met meerdere interne besluitvormingslagen. Het project kent een vaste aanneemsom en wordt uitgevoerd onder UAV-GC 2025. Een deel van de werkzaamheden wordt uitbesteed aan gespecialiseerde onderaannemers.\n\n" \
            "U gebruikt een AI‑gebaseerd beslissingsondersteunend systeem dat, op basis van projectkenmerken, een inschatting geeft van het risico op vertraging. Hieronder ziet u de ingevoerde projectkenmerken, gevolgd door de voorspelling en bijbehorende uitleg.\n\n"

        ),
        image_path=None,
        features=EXPERIMENT_SCENARIO_PRESET,
    ),
    Scenario(
        scenario_id="S1_en",
        title="Scenario 1 English",
        narrative_markdown=(
            """### Project scenario\n"""
            "You are involved as a project manager in a medium-sized utility project that is in the preparation phase, just before the start of the execution phase.\n\n"
            "The project involves the construction of a multifunctional building in an urban environment. The client is a professional organization with multiple internal decision-making levels. The project has a fixed contract value and will be executed under UAV-GC 2025. Some of the work will be subcontracted to specialized subcontractors.\n\n" \
            "You are using an AI-based decision-support system that, based on project characteristics, provides an estimate of the risk of project delay. Below you will see the entered project characteristics, followed by the prediction and corresponding explanation.\n\n"

        ),
        image_path=None,
        features=EXPERIMENT_SCENARIO_PRESET,
    )
]

XAI_DIR = Path("XAI")

def xai_path(condition: str, language: str = "nl") -> Path:
    base = XAI_DIR

    suffix = "_en" if language == "en" else ""

    if condition == "SHAP":
        return base / f"shap{suffix}.png"
    if condition == "Regels":
        return base / f"rules{suffix}.txt"
    if condition == "Tegenfeitelijk":
        return base / f"cf{suffix}.csv"
    if condition == "Surrogaatmodel (beslisboom)":
        return base / f"surrogate{suffix}.png"
    return base / ""
