import pyvisa as visa
import numpy as np
import time
from datetime import datetime, date
import matplotlib.pyplot as plt
import toolkit as tk
import pandas as pd

rm = visa.ResourceManager('')
# Print list of instruments detected on USB bus
print("*** USB Instruments Detected ***")
for resource in rm.list_resources():
    try :
        instrument=rm.open_resource(resource)
        ID = instrument.query("*IDN?")
    except :
        print("    ", "Resource does not respond to IDN query")
    print("\n  ", ID[:-1])  # the "-1 is to eliminate unneeded white space


# Some instruments are invisible to subsequent instrument.query, so reset
#   resource manager and then reopen it.
rm.close()
rm = visa.ResourceManager('')
try :
#	Search for scope on bus and, if found, open it for control
    for resource in rm.list_resources():
    # Check for Keysight DSOX1204G scope
        print(resource)
        if resource[6:20] == "0x2A8D::0x0396" :
            scope = rm.open_resource(resource)
            print("\n  ", scope.query("*IDN?"))
except :
    print ("Keysight DSOX1204G Oscilloscope either not detected or an error",
           "occurred.")

# Clear Scope Status and ReSeT scope to default setting
scope.write("*CLS;*RST")
# Wait a couple of seconds for reset to complete
time.sleep(2)