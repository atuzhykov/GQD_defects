# config.py
import os

molecules_data = {
    "QD_4": {
        "path": os.path.join("data", "QD_4.mol"),
        "cell": 15
    },
    "QD_6": {
        "path": os.path.join("data", "QD_6.mol"),
        "cell": 25
    },
    "QD_7": {
        "path": os.path.join("data", "QD_7.mol"),
        "cell": 30
    },

    "QD_8": {
        "path": os.path.join("data", "QD_9.mol"),
        "cell": 37
    },

    "QD_9": {
        "path": os.path.join("data", "QD_9.mol"),
        "cell": 39
    },

    "GQD_HEXAGON_3_3": {
        "path": os.path.join("data", "GQD_HEXAGON_3_3.mol"),
        "cell": 39
    },

    "GQD_HEXAGON_8_8": {
        "path": os.path.join("data", "GQD_HEXAGON_8_8.mol"),
        "cell": 62
    },
    "GQD_HEXAGON_3_3_func": {
        "path": os.path.join("data", "GQD_HEXAGON_3_3_func.mol"),
        "cell": 62
    },
    "GQD_HEXAGON_10_10_func": {
        "path": os.path.join("data", "GQD_HEXAGON_10_10_func.mol"),
        "cell": 62
    },

    "elementary": {
        "path": os.path.join("data", "elementary.mol"),
        "cell": 62
    }
}
