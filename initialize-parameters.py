import json

params = dict(
    # On MCC DAQ device
    VMON_AI_CHANNEL = 0,  # voltage monitor channel
    IMON_AI_CHANNEL = 1,  # current monitor channel
    PIF_AI_CHANNEL = 2,   # IF power input channel
    VCTRL_N_CHANNEL = 0,  # Control voltage negative channel
    VCTRL_P_CHANNEL = 1,  # Control voltage positive channel
    # # On SIS bias board
    VMON = dict(
        GAIN = -100, 
        OFFSET = 0,
        ),
    IMON = dict(
        GAIN = 1500, 
        OFFSET = 0,
        ),
)

with open('params.json', 'w') as fout:
    json.dump(params, fout, indent=4)
