
def upgrade_file(path):
    with open(path, "+tw") as f:
        f.write("# Sorry, the content is removed.")
        f.write("\n# Please ask Mike for the content.")

import numpy as np

def get_square(x):
    return np.square(x)