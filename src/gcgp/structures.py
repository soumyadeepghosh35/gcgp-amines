"""Structure identifiers for molecules, independent of how the SMILES is written.

Two SMILES strings can describe the same molecule (for example
``CNC1=CC=CC=C1`` and ``CNc1ccccc1``). The standard InChIKey computed by RDKit
is the same for both, so it is used as the identity of a structure.
"""

STRUCTURE_LEVELS = ("inchikey", "connectivity")


def structureKeys(smilesList, level="inchikey"):
    """Structure key and RDKit canonical SMILES for each SMILES string.

    ``level="inchikey"`` uses the full standard InChIKey, so stereoisomers that
    are written with stereochemistry stay separate. ``level="connectivity"``
    uses only its first block, so all stereoisomers of a skeleton share a key.
    A SMILES that RDKit cannot parse keeps its own text as the key (prefixed
    with ``UNPARSED:``), so it is never merged with another molecule.
    """
    from rdkit import Chem, RDLogger

    if level not in STRUCTURE_LEVELS:
        raise ValueError(f"structureLevel must be one of {STRUCTURE_LEVELS}.")
    RDLogger.DisableLog("rdApp.*")
    keys, canonical = [], []
    for smiles in smilesList:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            keys.append(f"UNPARSED:{smiles}")
            canonical.append(smiles)
            continue
        inchiKey = Chem.MolToInchiKey(mol)
        keys.append(inchiKey if level == "inchikey" else inchiKey.split("-")[0])
        canonical.append(Chem.MolToSmiles(mol))
    return keys, canonical
