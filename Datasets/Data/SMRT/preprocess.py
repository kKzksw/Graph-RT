import math
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

DATA_PATH = Path("Datasets/Data/SMRT")
SOURCE_PATH = DATA_PATH / "source"
OUTPUT_PATH = DATA_PATH / "all.csv"

RDLogger.DisableLog("rdApp.*")


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def canonicalize_smiles(smiles=None, inchi=None):
    molecule = None
    if not is_missing(smiles):
        molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None and not is_missing(inchi):
        molecule = Chem.MolFromInchi(str(inchi))
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True)


def build_standard_frame(
    frame, rt_column, rt_scale=1.0, smiles_column=None, inchi_column=None
):
    frame = frame.loc[
        :,
        [column for column in frame.columns if not str(column).startswith("Unnamed:")],
    ]

    rt_values = frame[rt_column].tolist()
    smiles_values = (
        frame[smiles_column].tolist()
        if smiles_column is not None
        else [None] * len(frame)
    )
    inchi_values = (
        frame[inchi_column].tolist()
        if inchi_column is not None
        else [None] * len(frame)
    )

    rows = []
    invalid_rows = 0
    for rt_value, smiles_value, inchi_value in zip(
        rt_values, smiles_values, inchi_values
    ):
        if is_missing(rt_value):
            invalid_rows += 1
            continue

        canonical_smiles = canonicalize_smiles(smiles=smiles_value, inchi=inchi_value)
        if canonical_smiles == "":
            invalid_rows += 1
            continue

        rows.append(
            {"rt_time": float(rt_value) * float(rt_scale), "smiles": canonical_smiles}
        )

    return pd.DataFrame(rows, columns=["rt_time", "smiles"]), invalid_rows


def prepare(force=False):
    if OUTPUT_PATH.exists() and not force:
        frame = pd.read_csv(OUTPUT_PATH)
        return {"dataset": "SMRT", "size": len(frame), "invalid_rows": 0}

    train_frame = pd.read_csv(SOURCE_PATH / "SMRT_train_set.txt", sep="\t")
    test_frame = pd.read_csv(SOURCE_PATH / "SMRT_test_set.txt", sep="\t")
    merged_frame = pd.concat([train_frame, test_frame], ignore_index=True)

    output_frame, invalid_rows = build_standard_frame(
        merged_frame,
        rt_column="RT",
        rt_scale=1.0,
        smiles_column="smiles",
    )
    output_frame.to_csv(OUTPUT_PATH, index=False)
    return {"dataset": "SMRT", "size": len(output_frame), "invalid_rows": invalid_rows}


if __name__ == "__main__":
    print(prepare())
