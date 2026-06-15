from __future__ import annotations

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
    "PRIMINS2",
    "PERSDOC3",
    "PREGNANT",
]

LIFESTYLE_COLS = [
    "ECIGNOW3",
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
    "LASTDEN4",
    "RMVTETH4",
    "CVDINFR4",
    "CHCOCNC1",
    "CHCCOPD3",
    "ADDEPEV3",
    "CHCKDNY2",
    "HAVARTH4",
    "DIABETE4",
    "DIABAGE4",
    "DEAF",
    "BLIND",
    "DECIDE",
    "DIFFWALK",
    "DIFFDRES",
    "DIFFALON",
    "HADHYST2",
    "FLUSHOT7",
    "PNEUVAC4",
    "PDIABTS1",
    "PREDIAB2",
    "DIABTYPE",
    "INSULIN1",
    "EYEEXAM1",
    "DIABEYE1",
    "DIABEDU1",
    "FEETSORE",
    "CNCRDIFF",
    "CNCRAGE",
    "CNCRTYP2",
    "CSRVTRT3",
    "CSRVPAIN",
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


def build_feature_datasets(df: pd.DataFrame, target_col: str) -> dict[str, pd.DataFrame]:
    feature_columns = [column for column in df.columns if column != target_col]

    lifestyle_feature_cols = _ordered(feature_columns, LIFESTYLE_COLS + SHARED_COLS)
    clinical_feature_cols = _ordered(feature_columns, CLINICAL_COLS + SHARED_COLS)

    return {
        "lifestyle": df[lifestyle_feature_cols + [target_col]],
        "clinical": df[clinical_feature_cols + [target_col]],
        "combined": df[feature_columns + [target_col]],
    }
