from __future__ import annotations

from collections import Counter

import pandas as pd


SHARED_COLS = [
    "_STATE",
    "MARITAL",
    "EDUCA",
    "VETERAN3",
    "EMPLOY1",
    "INCOME3",
    "SOMALE",
    "SOFEMALE",
    "_URBSTAT",
    "_RACE",
    "_SEX",
    "_AGE80",
    "GENHLTH",
    "PHYSHLTH",
    "MENTHLTH",
    "_BMI5",
]


LIFESTYLE_COLS = [
    "ECIGNOW3",
    "HIVRISK5",
    "ACEDEPRS",
    "ACEDRINK",
    "ACEDRUGS",
    "ACEPRISN",
    "ACEDIVRC",
    "ACEPUNCH",
    "ACEHURT1",
    "ACESWEAR",
    "ACETOUCH",
    "ACETTHEM",
    "ACEHVSEX",
    "ACEADSAF",
    "ACEADNED",
    "LSATISFY",
    "EMTSUPRT",
    "SDLONELY",
    "SDHEMPLY",
    "FOODSTMP",
    "SDHFOOD1",
    "SDHBILLS",
    "SDHUTILS",
    "SDHTRNSP",
    "HOWSAFE1",
    "MARIJAN1",
    "SSBSUGR2",
    "SSBFRUT3",
    "_TOTINDA",
    "_SMOKER3",
    "LCSFIRST",
    "LCSLAST_",
    "LCSNUMC_",
    "_LCSYSMK",
    "_LCSYQTS",
    "_DRNKWK3",
]


CLINICAL_COLS = [
    "PRIMINS2",
    "PERSDOC3",
    "CHECKUP1",
    "LASTDEN4",
    "RMVTETH4",
    "CVDINFR4",
    "CHCSCNC1",
    "CHCOCNC1",
    "CHCCOPD3",
    "ADDEPEV3",
    "CHCKDNY2",
    "HAVARTH4",
    "DIABETE4",
    "DIABAGE4",
    "PREGNANT",
    "DEAF",
    "BLIND",
    "DECIDE",
    "DIFFWALK",
    "DIFFDRES",
    "DIFFALON",
    "HADMAM",
    "HOWLONG",
    "CERVSCRN",
    "CRVCLCNC",
    "CRVCLPAP",
    "CRVCLHPV",
    "HADHYST2",
    "COLNSIGM",
    "COLNTES1",
    "SIGMTES1",
    "COLNCNCR",
    "STOLTEST",
    "FLUSHOT7",
    "PNEUVAC4",
    "HIVTST7",
    "PDIABTS1",
    "PREDIAB2",
    "DIABTYPE",
    "INSULIN1",
    "EYEEXAM1",
    "DIABEYE1",
    "DIABEDU1",
    "FEETSORE",
    "SHINGLE2",
    "HPVADVC4",
    "TETANUS1",
    "CNCRDIFF",
    "CNCRAGE",
    "CNCRTYP2",
    "CSRVTRT3",
    "CSRVPAIN",
    "PSATEST1",
    "CIMEMLO1",
    "CDWORRY",
    "CDDISCU1",
    "CDHOUS1",
    "CDSOCIA1",
    "_MICHD",
    "_ASTHMS1",
    "_LCSELIG",
    "_LCSCTSN",
]


def _ordered(feature_columns: list[str], selected_columns: list[str]) -> list[str]:
    selected = set(selected_columns)
    return [column for column in feature_columns if column in selected]


def _validate_feature_groups(feature_columns: list[str]) -> None:
    expected = set(feature_columns)
    grouped = SHARED_COLS + LIFESTYLE_COLS + CLINICAL_COLS
    grouped_set = set(grouped)

    duplicate_cols = sorted(column for column, count in Counter(grouped).items() if count > 1)
    missing_cols = sorted(grouped_set - expected)
    unassigned_cols = sorted(expected - grouped_set)

    errors = []
    if duplicate_cols:
        errors.append(f"assigned to multiple groups: {duplicate_cols}")
    if missing_cols:
        errors.append(f"not present in dataset: {missing_cols}")
    if unassigned_cols:
        errors.append(f"not assigned to any feature group: {unassigned_cols}")

    if errors:
        raise ValueError("Invalid feature group split; " + "; ".join(errors))


def build_feature_datasets(df: pd.DataFrame, target_col: str) -> dict[str, pd.DataFrame]:
    feature_columns = [column for column in df.columns if column != target_col]
    _validate_feature_groups(feature_columns)

    lifestyle_feature_cols = _ordered(feature_columns, LIFESTYLE_COLS + SHARED_COLS)
    clinical_feature_cols = _ordered(feature_columns, CLINICAL_COLS + SHARED_COLS)

    return {
        "lifestyle": df[lifestyle_feature_cols + [target_col]],
        "clinical": df[clinical_feature_cols + [target_col]],
        "combined": df[feature_columns + [target_col]],
    }
