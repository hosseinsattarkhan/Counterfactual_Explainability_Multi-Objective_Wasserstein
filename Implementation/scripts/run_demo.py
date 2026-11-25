import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from counterfactuals.prototype import minimum_preference_change,uniform_gap,worst_case_gap
# Two decisions, three scenarios, two objectives. Target 0 is minimax optimal
# for w0, but is NOT uniformly optimal for any preference weight.
costs=np.array([
 [[0.,8.],[8.,0.],[4.,4.]],
 [[3.,3.],[3.,3.],[9.,9.]],
])
w0=np.array([.8,.2])
result={'costs':costs.tolist(),'initial_weights':w0.tolist(),
        'target':0,'worst_case':minimum_preference_change(costs,0,w0,'worst_case'),
        'uniform':minimum_preference_change(costs,0,w0,'uniform')}
print(json.dumps(result,indent=2))
out=Path(__file__).resolve().parents[2]/'outputs'/'prototype_demo.json'
out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(result,indent=2))
