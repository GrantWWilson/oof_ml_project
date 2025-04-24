# oof_ml/utils/zernike_utils.py
"""
Utility functions and constants related to Zernike parameters for LMT OOF.
"""

ZERNIKE_TEMPLATE = [
    (0,  0, "BIAS",     0.00),
    (1,  1, "TILT_H",   None),  # None => fill from LHS or other source
    (2,  1, "TILT_V",   None),
    (3,  0, "FOCUS",    0.00),
    (4,  1, "AST_V",    None),
    (5,  1, "AST_O",    None),
    (6,  1, "COMA_H",   None),
    (7,  1, "COMA_V",   None),
    (8,  1, "TRE_O",    None),
    (9,  1, "TRE_V",    None),
    (10, 1, "SPH",      None),
    (11, 0, "2AST_V",   0.00),
    (12, 0, "2AST_O",   0.00),
    (13, 0, "TET_V",    0.00),
    (14, 0, "TET_O",    0.00),
    (15, 0, "2COMA_H",  0.00),
    (16, 0, "2COMA_V",  0.00),
    (17, 0, "2TRE_O",   0.00),
    (18, 0, "2TRE_V",   0.00),
    (19, 0, "PEN_O",    0.00),
    (20, 0, "PEN_V",    0.00),
    (21, 0, "2SPH",     0.00),
    (22, 0, "3AST_V",   0.00),
    (23, 0, "3AST_O",   0.00),
    (24, 0, "2TET_V",   0.00),
    (25, 0, "2TET_O",   0.00),
    (26, 0, "HEX_V",    0.00),
    (27, 0, "HEX_O",    0.00),
    (28, 0, "3COMA_H",  0.00),
    (29, 0, "3COMA_V",  0.00),
    (30, 0, "3TRE_O",   0.00),
    (31, 0, "3TRE_V",   0.00),
    (32, 0, "2PEN_O",   0.00),
    (33, 0, "2PEN_V",   0.00),
    (34, 0, "HEPT_O",   0.00),
    (35, 0, "HEPT_V",   0.00),
    (36, 0, "3SPH",     0.00),
    (37, 0, "4AST_V",   0.00),
    (38, 0, "4AST_O",   0.00),
    (39, 0, "3TET_V",   0.00),
    (40, 0, "3TET_O",   0.00),
    (41, 0, "2HEX_V",   0.00),
    (42, 0, "2HEX_O",   0.00),
    (43, 0, "OCT_V",    0.00),
    (44, 0, "OCT_O",    0.00),
]


def write_zernike_dat_file(output_path, param_dict):
    """
    Writes zernike.dat in the standard format for LMT OOF analysis,
    using the globally defined ZERNIKE_TEMPLATE.

    :param output_path:  Path/filename (e.g. "my_output_dir/zernike.dat")
    :param param_dict:   Dictionary that supplies floating values for
                         the Zernike terms that are labeled None in the template.
                         e.g. param_dict["TILT_H"] = 123.45
    """
    with open(output_path, 'w') as f:
        for (index, flag, name, default_value) in ZERNIKE_TEMPLATE:
            if default_value is None:
                if name not in param_dict:
                    raise KeyError(f"param_dict missing an entry for '{name}'")
                val = param_dict[name]
            else:
                val = default_value

            line = f"{index:<2}\t{flag:<2}\t{name:<8}\t{val:7.2f}\n"
            f.write(line)
