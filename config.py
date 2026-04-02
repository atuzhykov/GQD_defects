# config.py
# Cell sizes follow the rule: cell >= max_molecular_extent + 20 Å
# (10 Å vacuum on each side, sufficient for both GPAW DFT and SevenNet ML potential)
import os

molecules_data = {
    "QD_4": {
        "path": os.path.join("data", "QD_4.mol"),
        "cell": 35   # max_extent=13.4 A
    },
    "QD_6": {
        "path": os.path.join("data", "QD_6.mol"),
        "cell": 40   # max_extent=17.6 A
    },
    "QD_7": {
        "path": os.path.join("data", "QD_7.mol"),
        "cell": 45   # max_extent=21.8 A
    },
    "QD_8": {
        "path": os.path.join("data", "QD_9.mol"),
        "cell": 55   # max_extent=34.4 A
    },
    "QD_9": {
        "path": os.path.join("data", "QD_9.mol"),
        "cell": 55   # max_extent=34.4 A
    },
    #
    # "GQD_HEXAGON_3_3": {
    #     "path": os.path.join("data", "GQD_HEXAGON_3_3.mol"),
    #     "cell": 39   # max_extent=14.0 A
    # },
    #
    # "GQD_HEXAGON_8_8": {
    #     "path": os.path.join("data", "GQD_HEXAGON_8_8.mol"),
    #     "cell": 62
    # },
    # "GQD_HEXAGON_3_3_func": {
    #     "path": os.path.join("data", "GQD_HEXAGON_3_3_func.mol"),
    #     "cell": 62
    # },
    # "GQD_HEXAGON_10_10_func": {
    #     "path": os.path.join("data", "GQD_HEXAGON_10_10_func.mol"),
    #     "cell": 62
    # },
    #
    # "GQD_TRIANGLE_3": {
    #     "path": os.path.join("data", "GQD_TRIANGLE_3.mol"),
    #     "cell": 30   # max_extent=9.3 A
    # },
    # "GQD_TRIANGLE_2_2": {
    #     "path": os.path.join("data", "GQD_TRIANGLE_2_2.mol"),
    #     "cell": 30   # max_extent=7.1 A
    # },
    # "GQD_HEX_2_2_OH": {
    #     "path": os.path.join("data", "GQD_HEX_2_2_OH.mol"),
    #     "cell": 30   # max_extent=9.8 A
    # },
    # "GQD_HEX_2_2_NH2": {
    #     "path": os.path.join("data", "GQD_HEX_2_2_NH2.mol"),
    #     "cell": 35   # max_extent=10.1 A
    # },
    #
    "GQD_HEX_2_2": {
        "path": os.path.join("data", "GQD_HEX_2_2.mol"),
        "cell": 30   # max_extent=9.2 A
    },
    "GQD_HEX_3_3": {
        "path": os.path.join("data", "GQD_HEXAGON_3_3.mol"),
        "cell": 39   # max_extent=14.0 A (39 > required 35)
    },
    "Coronene": {
        "path": os.path.join("data", "Coronene.mol"),
        "cell": 30   # max_extent=9.5 A
    },
    "vacancy_atom_3": {
        "path": os.path.join("data", "vacancy_atom_3.xyz"),
        "cell": 25   # max_extent=5.0 A
    },
    "input_pyrene_C16H10": {
        "path": os.path.join("data", "input_pyrene_C16H10.mol"),
        "cell": 30   # max_extent=8.0 A
    },
}
