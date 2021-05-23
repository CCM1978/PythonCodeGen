import os
import time

for dirpath, dirnames, filenames in os.walk("repos/python"):

    for f in filenames:
        full_path = os.path.join(dirpath, f)
        #print(full_path)

        if full_path.endswith(".py"):
            #print(f"Keeping {full_path}")
            pass
        else:
            os.remove(full_path)