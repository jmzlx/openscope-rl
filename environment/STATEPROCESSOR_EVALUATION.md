# StateProcessor Evaluation for New Fields

## Summary

After adding new ATC-critical fields to the optimal extraction script, we evaluated whether StateProcessor needs updates.

## Decision: Keep New Fields as Metadata

**Conclusion**: New fields should remain as **metadata** and **NOT** be added to the core observation space.

## Rationale

### Current State
- StateProcessor extracts exactly **14 features** per aircraft
- Models are trained on this fixed feature dimension
- All existing saved models expect 14 features

### New Fields Added
1. `windComponents` - dict with cross/head wind
2. `flightPhase` - string (APRON, TAXI, CLIMB, etc.)
3. `nextWaypoint` - string waypoint name
4. `currentWaypoint` - string waypoint name
5. `flightPlanAltitude` - number
6. `flightPlanRoute` - string route
7. `hasApproachClearance` - boolean
8. `isOnFinal` - boolean
9. `isEstablishedOnGlidepath` - boolean

### Reasons to Keep as Metadata

1. **Backward Compatibility**
   - Existing models would break with feature dimension change
   - No need to retrain all models
   - Gradual migration possible

2. **Feature Dimension Constraints**
   - Wind components are dicts (not single floats)
   - Flight phase is categorical (would need encoding)
   - Waypoint names are strings (would need embedding)
   - Flight plan route is variable-length string
   
   Converting these to the 14-feature format would require:
   - Embeddings or one-hot encodings
   - Loss of information or dimension explosion
   - Complex preprocessing

3. **Usage Patterns**
   - These fields are useful for **action masking** and **decision rules**
   - Not necessarily needed in neural network input
   - Can be accessed directly from raw state when needed

4. **Optional Usage**
   - Models can choose to use metadata fields
   - Some models might not need them
   - Flexibility for different architectures

## Implementation Approach

### Current Flow (Unchanged)
```
Raw State → StateProcessor → Observation (14 features)
                              ↓
                           Models use 14 features
```

### Enhanced Flow (New Fields Available)
```
Raw State → StateProcessor → Observation (14 features) + Metadata (new fields)
                              ↓
                           Models can use:
                           - 14 features (core)
                           - Metadata fields (optional, via info dict)
```

### Where Metadata is Available

The new fields are available in:
1. **Raw state** from `extract_optimal_game_state()` - aircraft dicts have all fields
2. **Info dict** - can be passed through environment `step()` and `reset()` info
3. **Training data** - if using TrainingDataCollector with raw states

### Usage Examples

**Action Masking:**
```python
# In action masking or rule-based logic
if ac.get('hasApproachClearance'):
    # Don't allow approach clearance command
    mask[APPROACH_COMMAND] = 0
```

**Decision Logic:**
```python
# In reward calculation or heuristics
if ac.get('flightPhase') == 'APPROACH' and ac.get('isOnFinal'):
    # Adjust reward for final approach
    reward += approach_bonus
```

**Model Training (if needed):**
```python
# Models can optionally use metadata
metadata = {
    'wind': ac.get('windComponents'),
    'phase': ac.get('flightPhase'),
    'clearance': ac.get('hasApproachClearance'),
}
# Use in custom model architectures
```

## Recommendations

1. **Keep StateProcessor as-is** - 14 features remain the core
2. **Pass metadata via info dict** - Make new fields available in environment info
3. **Document metadata usage** - Show examples of how to use new fields
4. **Future consideration** - If models need these fields, consider:
   - Separate metadata encoder
   - Optional feature extension
   - Model-specific preprocessing

## Files Modified

- ✅ `environment/constants.py` - Enhanced extraction script
- ✅ `environment/utils.py` - Updated docstring
- ❌ `environment/state_processor.py` - **No changes needed**
- ❌ Model configs - **No changes needed** (feature dim stays 14)

## Conclusion

New fields are successfully extracted and available as metadata. StateProcessor remains unchanged, maintaining backward compatibility while providing rich ATC-critical data for advanced decision-making and action masking.

