# Missing Data Analysis - OpenScope Source Code Research

## Executive Summary

After comprehensive analysis of OpenScope source code and comparing to real ATC needs, we've identified and **implemented** extraction for all critical missing data points.

## Implementation Status

### ✅ **COMPLETED - All Critical Data Now Extracted**

All identified missing ATC-critical data has been added to the optimal extraction script.

---

## Previously Missing Data (Now Implemented)

### 1. **Wind Components** ✅ **IMPLEMENTED**

**Status:** ✅ Extracted in optimal script via `ac.getWindComponents()`

**Data:**
```javascript
windComponents: {
    cross: number,  // Crosswind in knots
    head: number    // Headwind/tailwind in knots (negative = tailwind)
}
```

**Use Cases:**
- Understand aircraft ground speed vs airspeed
- Predict wind impact on approach/landing
- Optimize runway assignments based on wind

---

### 2. **Flight Phase** ✅ **IMPLEMENTED**

**Status:** ✅ Extracted as `ac.flightPhase`

**Values:**
- `APRON`, `TAXI`, `WAITING`, `TAKEOFF`, `CLIMB`, `CRUISE`, `HOLD`, `DESCENT`, `APPROACH`, `LANDING`

**Use Cases:**
- Understand aircraft state and trajectory
- Apply phase-specific rules and constraints
- Optimize commands based on phase

---

### 3. **Flight Plan & Navigation Data** ✅ **IMPLEMENTED**

**Status:** ✅ All flight plan fields extracted

**Data Extracted:**
- `nextWaypoint` - Next waypoint name (string)
- `currentWaypoint` - Current waypoint name (string)
- `flightPlanAltitude` - Planned cruise altitude (number)
- `flightPlanRoute` - Full route string (string)
- `targetRunway` - Target runway name (string, critical for ILS commands)

**Operational State Extracted:**
- `isControllable` - Whether aircraft can receive commands (boolean, critical for action masking)
- `transponderCode` - Transponder/squawk code (string, for squawk commands)
- `groundTrack` - Actual track over ground (radians, accounts for wind drift)
- `trueAirspeed` - True airspeed (number, vs indicated airspeed)
- `climbRate` - Vertical speed/climb rate (number, ft/min)
- `distance` - Distance to airport/reference (number, nautical miles)

**Use Cases:**
- Understand aircraft intent and route
- Predict future positions
- Optimize clearances and route amendments

---

### 4. **Approach & Landing Status** ✅ **IMPLEMENTED**

**Status:** ✅ All approach/landing fields extracted

**Data Extracted:**
- `hasApproachClearance` - Whether cleared for approach (boolean)
- `isOnFinal` - On final approach segment (boolean)
- `isEstablishedOnGlidepath` - Established on glidepath (boolean)

**Use Cases:**
- Prevent duplicate approach clearances
- Understand landing sequence
- Apply landing-specific rules

---

### 5. **Game Events** ⚠️ **TESTED - Partially Accessible**

**Status:** ⚠️ Tested multiple access methods

**Test Results:**
1. **Direct GameController Access**: ❌ Not accessible (not on window)
2. **Window Property Search**: ❌ Not found
3. **Score Log Parsing**: ✅ Can parse DOM score log for recent events

**Available Events (from score log):**
- ARRIVAL, DEPARTURE
- SEPARATION_LOSS, NO_TAKEOFF_SEPARATION
- AIRSPACE_BUST, NOT_CLEARED_ON_ROUTE
- EXTREME_CROSSWIND_OPERATION, HIGH_CROSSWIND_OPERATION
- EXTREME_TAILWIND_OPERATION, HIGH_TAILWIND_OPERATION
- LOCALIZER_INTERCEPT_ABOVE_GLIDESLOPE
- ILLEGAL_APPROACH_CLEARANCE
- COLLISION

**Recommendation:**
- Use score log parsing for event tracking (available in test notebook)
- Score itself is sufficient for most RL training (aggregates all events)
- Events can be tracked via DOM parsing if granular metrics needed

---

## Complete Field List - Optimal Extraction

### Core Fields (14 - for StateProcessor)
1. `position` - [x, y] relative position
2. `altitude` - Current altitude
3. `heading` - Current heading
4. `speed` - Indicated airspeed
5. `groundSpeed` - Ground speed
6. `assignedAltitude` - MCP assigned altitude
7. `assignedHeading` - MCP assigned heading
8. `assignedSpeed` - MCP assigned speed
9. `isOnGround` - On ground boolean
10. `isTaxiing` - Taxiing boolean
11. `isEstablished` - Established on course
12. `category` - Arrival/departure
13. (derived: `is_arrival`)
14. (derived: `is_departure`)

### Additional ATC-Critical Fields (10 - navigation & approach)
15. `windComponents` - {cross, head} wind data
16. `flightPhase` - Current flight phase
17. `nextWaypoint` - Next waypoint name
18. `currentWaypoint` - Current waypoint name
19. `flightPlanAltitude` - Flight plan altitude
20. `flightPlanRoute` - Flight plan route string
21. `targetRunway` - Target runway name (critical for ILS commands and action masking)
22. `hasApproachClearance` - Approach clearance status
23. `isOnFinal` - On final approach
24. `isEstablishedOnGlidepath` - Established on glidepath

### Operational State Fields (6 - operational data)
25. `isControllable` - Whether aircraft can receive commands (critical for action masking)
26. `transponderCode` - Transponder/squawk code (for squawk commands)
27. `groundTrack` - Actual track over ground in radians (vs heading, accounts for wind)
28. `trueAirspeed` - True airspeed (vs indicated airspeed)
29. `climbRate` - Vertical speed/climb rate in ft/min
30. `distance` - Distance to airport/reference in nautical miles

---

## Real ATC Needs - Coverage Analysis

### ✅ **Position & Movement** - COMPLETE
- ✅ Position (relative to airport)
- ✅ Altitude (current and assigned)
- ✅ Heading (current and assigned)
- ✅ Speed (air, ground, and true airspeed)
- ✅ Ground track (actual path over ground, accounts for wind)
- ✅ Vertical speed/climb rate
- ✅ Distance to airport/reference
- ✅ Flight phase

### ✅ **Navigation & Intent** - COMPLETE
- ✅ Current waypoint
- ✅ Next waypoint
- ✅ Flight plan route
- ✅ Flight plan altitude
- ✅ Target runway

### ✅ **Weather & Environment** - COMPLETE
- ✅ Wind components (cross/head)
- ⚠️ Airport wind (not directly accessible, but available via aircraft references)
- ✅ Score (aggregate performance)

### ✅ **Operations & Clearances** - COMPLETE
- ✅ Approach clearance status
- ✅ Landing status (on final, on glidepath)
- ✅ Established status
- ✅ Category (arrival/departure)
- ✅ Ground operations (taxiing, on ground)
- ✅ Controllability status (isControllable)
- ✅ Transponder code (for squawk commands)

### ✅ **Conflicts & Safety** - COMPLETE
- ✅ Conflict pairs and distances
- ✅ Separation violations
- ✅ Conflict severity (hasConflict, hasViolation)

### ⚠️ **Performance Metrics** - PARTIAL
- ✅ Total score (aggregates all events)
- ⚠️ Individual event counts (requires DOM parsing)
- ✅ Violations tracked via conflicts

---

## Expert Player Needs - Coverage Analysis

### ✅ **All Real ATC Needs** - Covered

### ✅ **Game-Specific Metrics** - PARTIAL
- ✅ Score extracted
- ⚠️ Event breakdown (available via DOM parsing if needed)
- ✅ Aircraft count
- ✅ Conflict count

### ✅ **Strategic Information** - COMPLETE
- ✅ Flight plans (routes, altitudes)
- ✅ Waypoint progression
- ✅ Approach sequencing
- ✅ Wind conditions

---

## Data Accessibility Summary

### ✅ **Fully Accessible & Extracted:**
- All aircraft properties (via `window.aircraftController.aircraft.list`)
- Wind components (via `getWindComponents()` method)
- Flight plan data (via `fms` object)
- Approach status (via `pilot` object)
- Landing status (via methods)
- Conflicts (via `window.aircraftController.conflicts`)
- Score (via DOM)
- Game time (via injected function)

### ⚠️ **Partially Accessible:**
- Game events (via DOM score log parsing)
- Airport wind (via aircraft FMS references, not direct)
- Runway queue positions (not extracted, but could be added if needed)

### ❌ **Not Directly Accessible:**
- GameController (not on window - use score log parsing)
- TimeKeeper details (simulation rate, pause state - not critical)
- Traffic spawn rates (not needed for RL)

---

## Implementation Files

### Modified Files:
1. ✅ `environment/constants.py` - Enhanced `JS_GET_OPTIMAL_GAME_STATE_SCRIPT`
2. ✅ `environment/utils.py` - Updated `extract_optimal_game_state()` docstring
3. ✅ `notebooks/00_explore_openscope_api.ipynb` - Added game events tests
4. ✅ `notebooks/test_data_extraction.ipynb` - Created comprehensive test notebook
5. ✅ `environment/STATEPROCESSOR_EVALUATION.md` - Evaluation document

### Files NOT Modified (by design):
- ❌ `environment/state_processor.py` - Keeps 14 features for backward compatibility
- ❌ Model configs - Feature dimension remains 14
- ✅ New fields available as metadata via raw state

---

## Usage Guide

### Accessing New Fields

**In Environment:**
```python
state = extract_optimal_game_state(page)
for ac in state['aircraft']:
    wind = ac['windComponents']
    phase = ac['flightPhase']
    clearance = ac['hasApproachClearance']
    # Use for action masking, rules, etc.
```

**In Training Data:**
```python
# New fields included in raw state
episode.raw_states[step]['aircraft'][i]['windComponents']
episode.raw_states[step]['aircraft'][i]['flightPhase']
# etc.
```

**For Action Masking:**
```python
# Example: Don't allow approach clearance if already cleared
if aircraft['hasApproachClearance']:
    mask[APPROACH_COMMAND] = 0
```

---

## Testing Status

### ✅ Test Coverage:
1. ✅ Field extraction validated (test notebook)
2. ✅ Null handling tested
3. ✅ Data type validation
4. ✅ Performance benchmarks
5. ✅ Multiple aircraft scenarios
6. ✅ Game events access attempts

---

## Conclusion

**All critical ATC data is now extracted!** 

The optimal extraction script provides:
- ✅ All 14 core features for models
- ✅ 9 additional ATC-critical metadata fields
- ✅ Complete flight plan and navigation data
- ✅ Wind, phase, and clearance information
- ✅ Maximum efficiency (only extracts what's needed)

**Recommendation:** Use `extract_optimal_game_state()` for all production training data collection.

