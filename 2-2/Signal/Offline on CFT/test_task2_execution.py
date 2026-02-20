import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import sys

# specific mock to avoid blocking
def mock_show():
    print("plt.show() called")

plt.show = mock_show

# Now import/run the user code
try:
    import task2_solution
    print("Import successful")
    
    # task2_solution has a main block. run it.
    import runpy
    runpy.run_path('task2_solution.py', run_name='__main__')
    print("Execution successful")
except Exception as e:
    print(f"Execution failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
