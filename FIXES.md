# Bug Fixes and Known Issues

## Fixed Issues

### 1. IndexError: list index out of range (CRITICAL - Fixed)

**Error:**
```
IndexError: list index out of range
at environment/openscope_env.py:338
command_type = self.command_types[action["command_type"]]
```

**Cause:**
Mismatch between action space size in network definition and actual command types in environment.

- Network was configured with 7 command types (from config calculation)
- Environment only had 5 command types: `["altitude", "heading", "speed", "ils", "direct"]`

**Fix:**
- Updated `train.py` line 68: Changed from dynamic calculation to fixed `command_type: 5`
- Updated `evaluate.py` line 216: Same fix
- Now network and environment use consistent action space

**Files Modified:**
- `train.py`
- `evaluate.py`

---

## Known Warnings (Non-Critical)

### 1. Pydantic Field Attribute Warnings

**Warning:**
```
UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the Field() function...
UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the Field() function...
```

**Cause:**
Pydantic library (used by some dependencies) has changed its API in newer versions.

**Impact:** 
- **None** - These are just warnings from library internals
- Does not affect training functionality

**Solution:**
- Can be safely ignored
- Or suppress with: `export PYTHONWARNINGS="ignore::pydantic._internal._generate_schema.UnsupportedFieldAttributeWarning"`

### 2. PyTorch Transformer Warning

**Warning:**
```
UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
```

**Cause:**
PyTorch's TransformerEncoder disables nested tensor optimization when using pre-normalization (`norm_first=True`).

**Impact:**
- **Minor** - Slightly less efficient memory usage
- Training still works correctly
- Performance impact is negligible

**Solution:**
- Can be safely ignored
- This is expected behavior when using `norm_first=True` (which we do for better training stability)

---

## Configuration Note

### special_actions in config.yaml

The `special_actions` field in `config/training_config.yaml` is currently **not used** in the code. The command types are hardcoded in the environment as:

```python
command_types = ["altitude", "heading", "speed", "ils", "direct"]
```

If you want to extend the action space in the future:

1. Update `environment/openscope_env.py`:
   - Add new command to `self.command_types` list
   - Add handling in the `step()` method
   - Update action space size in `__init__`

2. Update `train.py` and `evaluate.py`:
   - Change `'command_type': 5` to the new number

3. Update `config/training_config.yaml`:
   - Modify `special_actions` list (for documentation)

---

## Verification

After these fixes, the training should run without errors. You should see:

```
Using device: cuda
Creating environment...
Creating agent...
Network parameters: 7,339,567

Starting training for 10000000 timesteps...
  0%|                          | 0/10000000 [00:00<?, ?it/s]
```

And it should progress without IndexError.

---

## Testing the Fix

To verify the fix works:

```bash
# Quick test
python -c "
from environment.openscope_env import OpenScopeEnv
from models.networks import ATCActorCritic
import torch

env = OpenScopeEnv(headless=True)
obs, _ = env.reset()

policy = ATCActorCritic()
obs_tensor = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}

action, _, _, _ = policy.get_action_and_value(obs_tensor)
print('✓ Action space compatible!')

obs, reward, term, trunc, info = env.step({
    k: v.cpu().numpy().item() if v.numel() == 1 else v.cpu().numpy() 
    for k, v in action.items()
})
print('✓ Environment step successful!')
env.close()
"
```

If both checks pass, training should work!

