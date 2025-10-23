# OpenScope Source Code Structure - Complete Analysis

## Directory Location
```
/home/jmzlx/Projects/atc-ai/openscope/
```

## Key Source Files Overview

### 1. Aircraft Representation
**Location**: `/home/jmzlx/Projects/atc-ai/openscope/src/assets/scripts/client/aircraft/`

#### AircraftModel.js
Main aircraft data model containing:

**Core Identifiers:**
- `id`: Unique identifier (auto-generated)
- `callsign`: Aircraft identifier (e.g., "AAL551")
- `airlineId`: Airline code (e.g., "AAL")
- `airlineCallsign`: Radio callsign (e.g., "American")
- `flightNumber`: Flight number only (e.g., "551")
- `transponderCode`: Transponder/squawk code (default 1200)

**Position & Movement:**
- `heading`: Magnetic heading in radians (0-2π)
- `altitude`: Altitude MSL in feet
- `speed`: Indicated Airspeed (IAS) in knots
- `groundSpeed`: Ground speed in knots
- `groundTrack`: Azimuth of ground movement in radians
- `trueAirspeed`: True Airspeed in knots
- `radial`: Bearing from airport center in radians
- `distance`: Distance from airport in km
- `relativePosition`: [x, y] position relative to airport
- `history`: Array of previous positions
- `trend`: Descent/Level/Climb indicator (-1, 0, or 1)

**Flight Information:**
- `origin`: Origin airport (for departures)
- `destination`: Destination airport (for arrivals)
- `flightPhase`: Current phase (APRON, TAXI, WAITING, TAKEOFF, CLIMB, CRUISE, HOLD, DESCENT, APPROACH, LANDING)
- `takeoffTime`: Game time of takeoff
- `taxi_start`: Start time of taxi
- `taxi_time`: Time spent taxiing (default 3 seconds)
- `rules`: Flight rules (IFR or VFR)

**Status:**
- `isControllable`: Whether in controlled airspace
- `isRemovable`: Whether safe to remove from simulation
- `isOnGround()`: Method to check if on ground
- `hit`: Whether aircraft has crashed
- `warning`: Warning status
- `projected`: Whether simulating future movements

**Conflicts:**
- `conflicts`: Dictionary of active conflicts
- `model`: AircraftTypeDefinitionModel with performance data

---

### 2. Commands & Control
**Location**: `/home/jmzlx/Projects/atc-ai/openscope/src/assets/scripts/client/commands/`

#### Main Aircraft Commands

**Altitude Control:**
- Command: `altitude` (aliases: `a`, `c`, `climb`, `d`, `descend`)
- Arguments: `altitude [expedite_flag]`
- Example: `AAL123 c 10000` (climb to 10,000 ft)
- Implementation: `AircraftCommander.runAltitude()` → `Pilot.maintainAltitude()`

**Heading Control:**
- Command: `heading` (aliases: `h`, `t`, `turn`, `fh`)
- Arguments: `heading [left|right]` or `heading degrees [left|right]` (incremental)
- Examples:
  - `AAL123 h 270` (fly heading 270)
  - `AAL123 t l 15` (turn left 15 degrees)
  - `AAL123 t r 30` (turn right 30 degrees)
- Implementation: `Pilot.maintainHeading()`

**Speed Control:**
- Command: `speed` (aliases: none standard)
- Arguments: `speed` (in knots)
- Example: `AAL123 speed 250` (maintain 250 knots)
- Implementation: `Pilot.maintainSpeed()`
- Validation: Aircraft performance limits checked

**Fly Present Heading:**
- Command: `flyPresentHeading` (aliases: `fph`)
- Maintains current magnetic heading
- Implementation: `Pilot.maintainPresentHeading()`

**Additional Commands:**
- `hold`: Enter holding pattern at specified fix
- `direct` (aliases: `dct`, `pd`): Direct to waypoint
- `ils` (aliases: `i`, `*`): Instrument Landing System approach
- `land`: Land at assigned runway
- `route`/`reroute`: Change flight route
- `sid`/`star`: Standard Instrument Departure/Arrival
- `taxi`: Request taxi clearance
- `takeoff` (aliases: `to`, `cto`): Request takeoff
- `cross` (aliases: `cr`, `x`): Cross specified waypoint at altitude
- `squawk`: Assign transponder code
- `delete` (aliases: `del`, `kill`): Remove aircraft from simulation

---

### 3. Collision Avoidance & Safety Rules
**Location**: `/home/jmzlx/Projects/atc-ai/openscope/src/assets/scripts/client/aircraft/AircraftConflict.js`
**Constants**: `/home/jmzlx/Projects/atc-ai/openscope/src/assets/scripts/client/constants/aircraftConstants.js`

#### Separation Standards (SEPARATION Object)

```javascript
SEPARATION = {
    MAX_KM: 14.816,                  // 8 nautical miles (max possible)
    STANDARD_LATERAL_KM: 5.556,      // 3 nautical miles (standard lateral minimum)
    VERTICAL_FT: 1000                // 1000 feet minimum vertical separation
}
```

#### Collision Detection

**Collision Threshold:**
- Lateral: < 0.05 km distance AND
- Vertical: < 160 feet altitude difference
- Both aircraft inside airspace
- Sets `aircraft.hit = true`

**Conflict Detection Process:**

1. **Exemption Rules:**
   - Aircraft below 990 feet AGL (above airport elevation): Ignored
   - Aircraft within 90 seconds of takeoff: Ignored
   - On ground: No conflicts checked

2. **Vertical Separation Check:**
   - If altitude difference >= 1000 feet: NO conflict/violation
   - Conflict/violation only when vertically close (< 1000 ft)

3. **Lateral Separation Minimum:**
   - Standard: 3 nm (5.556 km)
   - Parallel runways: May reduce based on runway relationship
   - Violation: Distance < lateral minimum
   - Conflict: Distance < lateral minimum + 1 nm (within warning zone)

4. **"Passing & Diverging" Exception Rules (FAA JO 7110.65):**
   - If heading difference >= 15 degrees (normal routes):
     - Opposite courses (>135° heading diff): OK if distance increasing
     - Same/crossing courses: Check ray intersection - OK if past convergence point
   - If heading difference >= 10 degrees on SID routes (reduced rule)
   - Exception cancels conflict/violation if conditions met

#### Conflict State Management

```javascript
conflicts = {
    proximityConflict: boolean,      // Warning zone violation
    proximityViolation: boolean,     // Safety zone violation
}

violations = {
    proximityViolation: boolean      // Active separation loss
}
```

Methods:
- `hasConflict()`: Any proximity conflicts active
- `hasViolation()`: Any separation violations active
- `checkCollision()`: Detect crashes
- `checkProximity()`: Calculate conflicts/violations
- `_recalculateLateralAndVerticalDistances()`: Update distances

---

### 4. Command Parser & Validator
**Location**: `/home/jmzlx/Projects/atc-ai/openscope/src/assets/scripts/client/commands/`

#### Command Definition Types (aircraftCommandDefinitions.js)

**Zero-Argument Commands:**
- `abort`, `clearedAsFiled`, `delete`, `flyPresentHeading`, `takeoff`
- `sayAltitude`, `sayHeading`, `saySpeed`, `sayRoute`
- System commands: `airac`, `auto`, `clear`, `pause`, `tutorial`

**Single-Argument Commands:**
- `airport`, `direct`, `expectArrivalRunway`, `ils`, `land`
- `moveDataBlock`, `reroute`, `route`, `sid`, `speed`, `star`
- `rate`: Special parsing to convert string → number

**Custom-Argument Commands:**
- `altitude`: Custom altitude parser with expedite support
- `heading`: Heading parser with direction/incremental support
- `cross`: Crossing altitude parser
- `fix`: Waypoint selection
- `hold`: Hold pattern parser
- `squawk`: Transponder code
- `taxi`: Optional argument
- `climbViaSid`/`descendViaStar`: Optional altitude

---

### 5. Aircraft Dynamics & Performance
**Location**: `/home/jmzlx/Projects/atc-ai/openscope/src/assets/scripts/client/constants/aircraftConstants.js`

#### Performance Constants (PERFORMANCE Object)

```javascript
PERFORMANCE = {
    // Turn and climb/descent rates
    TURN_RATE: 0.0523598776,              // 3 degrees per second
    TYPICAL_CLIMB_FACTOR: 0.7,            // 70% of max climb rate
    TYPICAL_DESCENT_FACTOR: 0.7,          // 70% of max descent rate
    DECELERATION_FACTOR_DUE_TO_GROUND_BRAKING: 3.5,  // 3.5x faster on ground
    
    // Approach establishment criteria
    MAXIMUM_DISTANCE_CONSIDERED_ESTABLISHED_ON_APPROACH_COURSE_NM: 0.0822894,  // ~500 feet
    MAXIMUM_ANGLE_CONSIDERED_ESTABLISHED_ON_APPROACH_COURSE: 0.0872665,        // ~5 degrees
    
    // Waypoint navigation
    MAXIMUM_DISTANCE_TO_FLY_BY_WAYPOINT_NM: 5,        // Fly-by radius
    MAXIMUM_DISTANCE_TO_PASS_WAYPOINT_NM: 0.5,        // Fly-over threshold
    
    // Descent/approach
    INSTRUMENT_APPROACH_MINIMUM_DESCENT_ALTITUDE: 200,  // MDA in feet AGL
    MAXIMUM_ALTITUDE_DIFFERENCE_CONSIDERED_ESTABLISHED_ON_GLIDEPATH: 100,  // ft
    
    // Stability requirements
    STABLE_APPROACH_TIME_SECONDS: 60,     // Time to reach Vref before landing
    
    // Takeoff
    TAKEOFF_TURN_ALTITUDE: 400,           // Altitude to begin turn (ft)
    
    // Standard pressure
    DEFAULT_ALTIMETER_IN_INHG: 29.92      // Standard setting (inHg)
}
```

---

### 6. Flight Phases
**Location**: `/home/jmzlx/Projects/atc-ai/openscope/src/assets/scripts/client/constants/aircraftConstants.js`

```javascript
FLIGHT_PHASE = {
    APRON: 'APRON',              // Gate/parking position
    TAXI: 'TAXI',                // Taxiing to runway
    WAITING: 'WAITING',          // Ready for takeoff, awaiting clearance
    TAKEOFF: 'TAKEOFF',          // Airborne, climbing out
    CLIMB: 'CLIMB',              // Climbing to cruise altitude
    CRUISE: 'CRUISE',            // At assigned cruise altitude
    HOLD: 'HOLD',                // In holding pattern
    DESCENT: 'DESCENT',          // Descending from cruise
    APPROACH: 'APPROACH',        // On approach, outside final approach fix
    LANDING: 'LANDING'           // On final approach, within FAF
}
```

---

### 7. Flight Categories
```javascript
FLIGHT_CATEGORY = {
    ARRIVAL: 'arrival',
    DEPARTURE: 'departure',
    OVERFLIGHT: 'overflight'
}
```

---

### 8. Key Controller Classes

#### AircraftCommander (aircraftCommand/AircraftCommander.js)
- `runCommands()`: Execute array of commands on aircraft
- `run()`: Execute single command
- `runAltitude()`: Set altitude clearance
- `runHeading()`: Set heading clearance
- Handles deferred commands (takeoff commands executed last)
- Provides readback to UI

#### Pilot (Pilot/Pilot.js)
- `maintainAltitude()`: Implement altitude change
- `maintainHeading()`: Implement heading change
- `maintainSpeed()`: Implement speed change
- `maintainPresentHeading()`: Maintain current heading
- `applyArrivalProcedure()`: Load STAR
- `applyDepartureProcedure()`: Load SID
- `applyInstrumentApproach()`: Load approach
- Validates performance limits for speed commands

#### AircraftController (aircraft/AircraftController.js)
- Manages creation/destruction of aircraft
- Handles spawning and removal
- Updates all aircraft each frame
- Manages conflicts

#### AircraftTypeDefinitionModel (AircraftTypeDefinitionModel.js)
- Performance characteristics:
  - Cruise speed
  - Max climb/descent rates
  - Turn radius
  - Acceleration/deceleration
  - Min/max speeds for flight phases
- Method: `isAbleToMaintainSpeed(speed)`: Validates speed feasibility

---

## Critical Physics/Movement

### Heading
- Units: Radians (0 to 2π)
- Type: Magnetic heading
- Turn rate: ~3 degrees per second
- Affected by wind

### Altitude
- Units: Feet MSL (Mean Sea Level)
- Climb rate: ~70% of maximum
- Descent rate: ~70% of maximum
- Vertical separation minimum: 1000 feet

### Speed
- Units: Knots (Indicated Airspeed - IAS)
- Aircraft-specific limits enforced
- Ground speed calculated from IAS + wind

### Separation Rules Summary

**When to Check:**
- Both aircraft above 990 feet AGL
- Both past 90-second post-takeoff window
- Altitude difference < 1000 feet

**Violation Threshold:**
- Lateral distance < 3 nm (5.556 km)

**Warning Threshold:**
- Lateral distance < 3 nm + 1 nm = 4 nm (7.408 km)
- Exception: "Passing & Diverging" rule may cancel

**Collision Threshold:**
- Lateral distance < 0.05 km (~164 feet)
- Vertical distance < 160 feet

---

## Command Syntax Examples

```
# Altitude changes
AAL123 c 10000           # Climb to 10,000 feet
AAL123 d 5000            # Descend to 5,000 feet
AAL123 a 7000 expedite   # Altitude 7,000 expedite

# Heading changes
AAL123 h 270             # Fly heading 270
AAL123 t r 30            # Turn right 30 degrees
AAL123 t l 15            # Turn left 15 degrees
AAL123 fph               # Fly present heading

# Speed changes
AAL123 speed 250         # Maintain 250 knots

# Approaches
AAL123 ils 25L           # ILS approach runway 25L
AAL123 land              # Land at assigned runway

# Navigation
AAL123 direct BARKS      # Direct to waypoint BARKS
AAL123 cross BARKS 8000  # Cross BARKS at 8,000 feet
AAL123 hold BARKS        # Hold at BARKS

# Procedures
AAL123 sid KEEPR1        # Climb via SID
AAL123 star SUNOL2       # Descend via STAR

# Other
AAL123 route SUNOL KORD  # Route via waypoint to destination
AAL123 taxi              # Request taxi clearance
AAL123 to                # Request takeoff
AAL123 squawk 2401       # Transponder 2401
AAL123 delete            # Remove from simulation
```

---

## File References Summary

| Category | File | Purpose |
|----------|------|---------|
| Aircraft Model | `AircraftModel.js` | Core aircraft state |
| Commands | `AircraftCommander.js` | Command execution |
| Pilot | `Pilot/Pilot.js` | Altitude/heading/speed implementation |
| Conflicts | `AircraftConflict.js` | Collision/separation detection |
| Constants | `aircraftConstants.js` | Performance, separation, phases |
| Command Definitions | `aircraftCommandDefinitions.js` | Command syntax & validation |
| Command Map | `aircraftCommandMap.js` | Command routing |
| Type Definition | `AircraftTypeDefinitionModel.js` | Aircraft performance data |

---

## Key Integration Points for RL Training

1. **Observation Space**: Extract from `AircraftModel`:
   - Per-aircraft features: heading, altitude, speed, position, trend
   - Global features: airport elevation, wind
   - Relative positions between aircraft

2. **Action Space**: From command system:
   - Altitude: Any valid altitude per airport/aircraft
   - Heading: 0-360 degrees
   - Speed: Aircraft-specific min/max

3. **Safety Constraints**: From `AircraftConflict`:
   - Separation violations: distance < 3nm, vertical < 1000ft
   - Collision: distance < 0.05km, vertical < 160ft
   - Passing/Diverging exception logic

4. **Reward Signals**:
   - Successful separation maintenance
   - Smooth trajectories (minimize conflicts)
   - Efficient routing
   - Penalty for violations/collisions

