import time
start = time.time()
print("Importing...")
import pandas
import joblib
import sklearn
try:
    import tensorflow
    print("Tensorflow imported")
except ImportError:
    print("Tensorflow not found")

end = time.time()
print(f"Imports took {end - start} seconds")
