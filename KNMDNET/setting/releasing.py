import gc
import matplotlib.pyplot as plt
import os
def clear():
    for key, value in globals().keys():
        if callable(value) or value.__class__.__name__ == "module":
            continue
        del globals()[key]
def __clear_env():
    plt.close("all")
    os.system('os')
    gc.collect()
    for key in list(globals().keys()):
        if (not key.startswith("__")) and (key != "key"):
            globals().pop(key)
    del key
