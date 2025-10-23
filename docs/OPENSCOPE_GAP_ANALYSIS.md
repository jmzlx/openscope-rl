# OpenScope vs Self-Contained Environment - Gap Analysis

**Date:** 2025-10-15
**Purpose:** Identify differences between OpenScope browser-based ATC simulator and the self-contained Python RL training environment to enable accurate replication of game logic without external dependencies.

---

## Executive Summary

This gap analysis compares the full-featured OpenScope air traffic control simulator with the simplified self-contained Python environment created for reinforcement learning training. The objective is to identify missing features and logic that should be replicated to maintain training fidelity while eliminating browser/networking dependencies.

### Key Findings:
- **Current Implementation Coverage:** ~30% of OpenScope features
- **Critical Gaps:** Flight phases, approach procedures, scoring system, wind effects
- **Recommended Additions:** 15 high-priority features for RL training effectiveness

---

## 1. SCORING AND REWARD SYSTEM

### OpenScope Implementation

**Point Values:**
| Event | Points | Condition |
|-------|--------|-----------|
| ARRIVAL | +10 | Aircraft lands successfully and stops |
| DEPARTURE | +10 | Aircraft exits on assigned route |
| AIRSPACE_BUST | -200 | Arrival exits airspace without landing |
| COLLISION | -1000 | Aircraft within 0.05nm & <160ft vertical |
| SEPARATION_LOSS | -200 | Violation of 3nm lateral separation |
| NO_TAKEOFF_SEPARATION | -200 | Inadequate departure spacing |
| NOT_CLEARED_ON_ROUTE | -25 | Departure exits without route clearance |
| GO_AROUND | -50 | Missed approach initiated |
| EXTREME_TAILWIND_OPERATION | -75 | Tailwind >10 knots |
| HIGH_TAILWIND_OPERATION | -25 | Tailwind 5-10 knots |
| EXTREME_CROSSWIND_OPERATION | -15 | Crosswind >20 knots |
| HIGH_CROSSWIND_OPERATION | -5 | Crosswind 10-20 knots |
| ILLEGAL_APPROACH_CLEARANCE | -10 | Intercept angle >30 degrees |
| LOCALIZER_INTERCEPT_ABOVE_GLIDESLOPE | -10 | Approach above glideslope |

**Weighted Score Formula:**
```
scorePerHour = totalScore / hoursPlayed
```
Normalizes performance across different session lengths.

### Self-Contained Environment Implementation

**Current Rewards:**
- Command issued: +0.1
- Landing initiated: +0.5
- Successful landing: +20.0
- Conflict warning: -1.0
- Separation violation: -10.0

**Gaps:**
- ❌ No departure scoring
- ❌ No airspace bust detection
- ❌ No go-around penalty
- ❌ No wind-based penalties
- ❌ No approach procedure penalties
- ❌ No weighted score calculation
- ❌ No collision detection (separate from violations)
- ❌ No runway separation enforcement

**Priority: HIGH** - Reward shaping is critical for RL training effectiveness.

### Recommendations:
1. ✅ Add all 14 OpenScope scoring events
2. ✅ Implement weighted scoring
3. ✅ Add event tracking/logging
4. ✅ Separate collision (-1000) from violation (-200) penalties
5. ✅ Add wind component calculation and penalties

---

## 2. AIRCRAFT FLIGHT PHASES AND STATE MACHINE

### OpenScope Implementation

**11 Flight Phases:**
```
APRON → TAXI → WAITING → TAKEOFF → CLIMB → CRUISE → HOLD → DESCENT → APPROACH → LANDING → REMOVED
```

**Phase-Specific Behaviors:**
- **APRON/TAXI/WAITING:** Not visible on radar, ground operations
- **TAKEOFF:** Visible, accelerating, transitions at 400ft AGL
- **CLIMB:** Auto-climb to cruise altitude, 250kt speed limit <10k ft
- **CRUISE:** Level flight, awaits descent clearance
- **HOLD:** Airborne holding pattern at waypoint
- **DESCENT:** Active descent, can transition to approach
- **APPROACH:** Established on course, outside FAF (>5nm)
- **LANDING:** Within FAF (<5nm), glideslope tracking

**Automatic Transitions:**
- Altitude-based (TAKEOFF→CLIMB at 400ft)
- Course-based (DESCENT→APPROACH when established)
- Distance-based (APPROACH→LANDING at 5nm FAF)

### Self-Contained Environment Implementation

**Current State:**
- ✅ Position tracking (x, y)
- ✅ Altitude tracking
- ✅ Heading tracking
- ✅ Speed tracking
- ✅ Simple landing flag (`is_landing`)
- ✅ Runway assignment

**Gaps:**
- ❌ No flight phase state machine
- ❌ No automatic phase transitions
- ❌ No phase-specific command restrictions
- ❌ No ground operations (APRON/TAXI/WAITING)
- ❌ No holding patterns
- ❌ No approach procedures
- ❌ No altitude-dependent speed constraints (250kt <10k ft)

**Priority: HIGH** - Flight phases are essential for realistic ATC simulation.

### Recommendations:
1. ✅ Implement flight phase enum and state machine
2. ✅ Add automatic phase transitions based on altitude/distance
3. ✅ Restrict commands by phase (e.g., no ILS clearance in CRUISE)
4. 🟡 Consider simplified 6-phase model for initial implementation:
   - AIRBORNE_CRUISE (merges CLIMB/CRUISE/DESCENT)
   - APPROACH
   - LANDING
   - LANDING_COMPLETE
   - (Skip ground phases for RL simplicity)

---

## 3. COLLISION AVOIDANCE AND SEPARATION RULES

### OpenScope Implementation

**Separation Standards:**
```javascript
SEPARATION = {
    MAX_KM: 14.816,              // 8nm conflict detection range
    STANDARD_LATERAL_KM: 5.556,  // 3nm minimum separation
    VERTICAL_FT: 1000            // 1000 feet vertical minimum
}
```

**Conflict Types:**
- **Collision:** <0.05nm lateral AND <160ft vertical → -1000 points
- **Violation:** <3nm lateral AND <1000ft vertical → -200 points
- **Conflict:** <4nm lateral AND <1000ft vertical → warning only

**"Passing & Diverging" Exception (FAA JO 7110.65):**
- If heading difference ≥15 degrees (10° for SIDs):
  - Opposite courses (>135° difference): OK if distance increasing
  - Same/crossing courses: OK if past point of convergence
  - Cancels conflict/violation status

**Exemptions:**
- Either aircraft below 990ft AGL
- Either aircraft within 90 seconds of takeoff
- Either aircraft on ground

### Self-Contained Environment Implementation

**Current Implementation:**
```python
SEPARATION_LATERAL_NM = 3.0
SEPARATION_VERTICAL_FT = 1000.0
CONFLICT_WARNING_BUFFER_NM = 1.0

def check_conflict(self, other: 'Aircraft') -> Tuple[bool, bool]:
    lateral_dist = self.distance_to(other)
    vertical_sep = self.vertical_separation(other)

    if vertical_sep >= SEPARATION_VERTICAL_FT:
        return False, False

    is_violation = lateral_dist < SEPARATION_LATERAL_NM
    is_conflict = lateral_dist < (SEPARATION_LATERAL_NM + CONFLICT_WARNING_BUFFER_NM)

    return is_violation, is_conflict
```

**Gaps:**
- ❌ No collision detection (different from violation)
- ❌ No "passing & diverging" exception logic
- ❌ No heading-based divergence checks
- ❌ No exemptions for low-altitude/takeoff/ground aircraft
- ❌ No 8nm maximum conflict range optimization

**Priority: MEDIUM-HIGH** - Passing & diverging logic is important for realistic ATC.

### Recommendations:
1. ✅ Add collision detection (<0.05nm, <160ft) with -1000 penalty
2. ✅ Implement passing & diverging logic:
   - Calculate heading difference
   - Check if distance is increasing
   - Apply exemption if conditions met
3. ✅ Add low-altitude exemption (<990ft AGL)
4. 🟡 Add takeoff time tracking for 90-second exemption
5. 🟡 Optimize with 8nm bounding box check

---

## 4. COMMAND VALIDATION AND CONSTRAINTS

### OpenScope Implementation

**Altitude Constraints:**
- Aircraft ceiling (e.g., B737: 43,000ft)
- Airport minimum safe altitude (MSA)
- Airport ceiling (typically 10,000ft controlled airspace)
- Minimum descent altitude (MDA): Airport elevation + 200ft
- Restricted airspace floor/ceiling boundaries

**Speed Constraints:**
- Aircraft minimum speed (e.g., 130kt for B737)
- Aircraft maximum speed (e.g., 492kt for B737)
- Aircraft landing speed (e.g., 140kt)
- FAA rule: ≤250kt below 10,000ft MSL
- Phase-specific constraints (approach speed, etc.)

**Heading Constraints:**
- Valid range: 001-360 degrees
- Turn direction: left/right
- Incremental or absolute heading

**Command Prerequisites:**
- Approach clearance requires descent phase
- Heading commands cancel approach clearance
- Runway assignment requires aircraft be airborne
- Commands rejected if aircraft not controllable

### Self-Contained Environment Implementation

**Current Implementation:**
- ✅ Discrete altitude actions: 0-17 (0, 1000, 2000, ..., 17000ft)
- ✅ Discrete heading actions: 0-11 (0, 30, 60, ..., 330 degrees)
- ✅ Discrete speed actions: 0-7 (150, 180, 210, ..., 360 knots)
- ✅ No-op action supported

**Gaps:**
- ❌ No aircraft-specific performance limits
- ❌ No altitude validation (MSA, ceiling)
- ❌ No 250kt speed limit below 10,000ft
- ❌ No command prerequisite checks
- ❌ No phase-based command restrictions
- ❌ No approach clearance state tracking
- ❌ No validation error messages/feedback

**Priority: MEDIUM** - Command validation improves realism but may be relaxed for RL.

### Recommendations:
1. ✅ Add 250kt speed limit below 10,000ft
2. ✅ Add minimum/maximum altitude bounds (e.g., 0-17,000ft)
3. 🟡 Add approach clearance state (cancels with heading commands)
4. 🟡 Add validation feedback to observation space
5. ❌ Skip aircraft-specific limits initially (use generic limits)

---

## 5. RUNWAY OPERATIONS AND LANDING PROCEDURES

### OpenScope Implementation

**ILS Approach:**
- Localizer: 25km range, centerline guidance
- Glideslope: 3-degree descent path
- Final Approach Fix (FAF): 5nm from runway
- Minimum glideslope intercept altitude: Calculated from FAF, rounded to 100ft
- Course establishment tolerance: ≤500ft lateral, ≤5° heading
- Glideslope establishment tolerance: ≤100ft altitude difference

**Landing Sequence:**
```
DESCENT → (approach clearance + established) → APPROACH → (within 5nm FAF + on glideslope) → LANDING → (touchdown) → FULLSTOP
```

**Go-Around:**
- **Automatic:** If not on glideslope when entering LANDING phase
- **Manual:** "go around" command
- **Penalty:** -50 points
- **Missed approach altitude:** Runway elevation + 2000ft (rounded up)

**Wind Effects:**
- Wind increases with altitude (factor per foot)
- Crosswind/headwind components calculated per runway
- Penalties for excessive crosswind/tailwind operations

**Runway Separation:**
- Category-based spacing (3000ft, 4500ft, 6000ft)
- Same-runway departure spacing enforced
- Penalty: -200 points for insufficient spacing

### Self-Contained Environment Implementation

**Current Implementation:**
- ✅ 4 runway directions (09/27, 04/22 - two intersecting runways)
- ✅ Landing detection when altitude <100ft and distance <0.5nm to runway
- ✅ Successful landing reward: +20 points
- ✅ Runway assignment per aircraft

**Gaps:**
- ❌ No ILS approach procedures
- ❌ No localizer/glideslope tracking
- ❌ No FAF (5nm final approach fix)
- ❌ No course establishment checks
- ❌ No automatic go-around logic
- ❌ No manual go-around command
- ❌ No wind simulation or penalties
- ❌ No runway separation enforcement
- ❌ No approach clearance management
- ❌ No glideslope altitude calculation

**Priority: HIGH** - Approach procedures are central to ATC operations.

### Recommendations:
1. ✅ Add FAF marker at 5nm from runway
2. ✅ Add ILS approach clearance command
3. ✅ Implement course establishment checks (lateral & heading)
4. ✅ Add glideslope altitude calculation (3-degree path)
5. ✅ Implement automatic go-around if not on glideslope at FAF
6. ✅ Add go-around command and -50 penalty
7. 🟡 Add basic wind simulation (constant wind)
8. 🟡 Calculate crosswind/headwind components
9. 🟡 Add wind-based penalties
10. ❌ Skip category-based spacing initially (use simple time/distance)

---

## 6. AIRCRAFT DYNAMICS AND PHYSICS

### OpenScope Implementation

**Turn Dynamics:**
- Turn rate: 3 degrees per second (0.0523598776 radians/sec)
- Bank angle calculations (not exposed)
- Wind drift during turns

**Climb/Descent:**
- Aircraft-specific climb rate (e.g., B737: 2500 fpm)
- Aircraft-specific descent rate (e.g., B737: 3500 fpm)
- Typical factor: 70% of maximum rate
- Speed affects climb/descent performance

**Acceleration:**
- Aircraft-specific acceleration (e.g., 6 kt/s)
- Aircraft-specific deceleration (e.g., 3 kt/s)
- Ground braking: 3.5x faster deceleration
- Thrust limits by phase

**Position Updates:**
- Wind vector added to aircraft velocity
- Groundspeed = true airspeed + wind effect
- Ground track ≠ heading (wind drift)

### Self-Contained Environment Implementation

**Current Implementation:**
```python
TURN_RATE_DEG_PER_SEC = 3.0
FT_PER_SEC_CLIMB = 2000 / 60  # ~33.3 ft/s (2000 fpm)
FT_PER_SEC_DESCENT = 2000 / 60

def update(self, dt: float):
    # Heading update (3 deg/sec)
    max_turn = TURN_RATE_DEG_PER_SEC * dt

    # Altitude update
    climb_rate = FT_PER_SEC_CLIMB if alt_diff > 0 else -FT_PER_SEC_DESCENT

    # Speed update (instant - simplified)
    self.speed = self.target_speed

    # Position update (no wind)
    speed_nm_per_sec = self.speed / 3600.0
    self.x += speed_nm_per_sec * dt * np.sin(heading_rad)
    self.y += speed_nm_per_sec * dt * np.cos(heading_rad)
```

**Gaps:**
- ❌ No wind effects on position
- ❌ No groundspeed vs airspeed distinction
- ❌ No ground track vs heading distinction
- ❌ No acceleration/deceleration (instant speed change)
- ❌ No aircraft-specific performance variations
- ❌ No altitude-dependent performance (thinner air at altitude)
- ❌ No ground braking enhancement

**Priority: MEDIUM** - Wind and acceleration add realism but increase complexity.

### Recommendations:
1. ✅ Add wind vector to position updates
2. ✅ Separate groundspeed from airspeed
3. ✅ Calculate ground track with wind drift
4. 🟡 Add acceleration/deceleration dynamics (6 kt/s, 3 kt/s)
5. 🟡 Add ground braking (3.5x factor)
6. ❌ Skip aircraft-specific variations initially
7. ❌ Skip altitude-dependent performance initially

---

## 7. OBSERVATION SPACE

### OpenScope Game State

**Available Data:**
- Per-aircraft:
  - Position (x, y, altitude)
  - Velocity (speed, heading, groundSpeed, groundTrack)
  - Target values (target altitude, heading, speed)
  - Flight phase (11 phases)
  - Flight plan (origin, destination, route, waypoints)
  - Status flags (isControllable, isRemovable, hit, warning)
  - Conflict/violation state
  - Runway assignment
  - Distance/bearing to airport
  - Distance/bearing to next waypoint
  - Time to next waypoint
  - Transponder code
  - Category (arrival/departure/overflight)
- Global:
  - Current time
  - Score
  - Event counts (arrivals, departures, violations, etc.)
  - Active runway(s)
  - Wind (angle, speed)
  - Airport configuration

### Self-Contained Environment Observation

**Current Implementation:**
```python
observation_space = spaces.Dict({
    'aircraft': spaces.Box(
        shape=(max_aircraft, 14), dtype=np.float32
    ),  # Per-aircraft features
    'aircraft_mask': spaces.Box(
        shape=(max_aircraft,), dtype=bool
    ),  # Valid aircraft mask
    'conflict_matrix': spaces.Box(
        shape=(max_aircraft, max_aircraft), dtype=np.float32
    ),  # Pairwise conflicts
    'global_state': spaces.Box(
        shape=(4,), dtype=np.float32
    )  # Time, score, violations
})
```

**Per-Aircraft Features (14):**
1. x position (normalized)
2. y position (normalized)
3. altitude (normalized)
4. heading (normalized)
5. speed (normalized)
6. target altitude (normalized)
7. target heading (normalized)
8. target speed (normalized)
9. dx to nearest runway (normalized)
10. dy to nearest runway (normalized)
11. runway assigned (normalized)
12. is_landing flag
13. distance to airport (normalized)
14. bearing to airport (normalized)

**Global State Features (4):**
1. Number of aircraft (normalized)
2. Time elapsed (normalized)
3. Score (normalized)
4. Violations count (normalized)

**Gaps:**
- ❌ No flight phase information
- ❌ No groundspeed/ground track
- ❌ No approach clearance status
- ❌ No distance to FAF
- ❌ No glideslope deviation
- ❌ No wind information
- ❌ No aircraft category (arrival/departure)
- ❌ No route/waypoint information
- ❌ No time to FAF/runway
- ❌ No event counts (departures, landings, etc.)

**Priority: MEDIUM-HIGH** - Richer observations improve RL policy quality.

### Recommendations:
1. ✅ Add flight phase (one-hot encoded or integer)
2. ✅ Add approach clearance flag
3. ✅ Add distance to FAF
4. ✅ Add glideslope deviation (altitude error from ideal)
5. ✅ Add wind vector (angle, speed)
6. ✅ Add aircraft category (arrival=0, departure=1)
7. 🟡 Add event counters to global state
8. 🟡 Add groundspeed and ground track
9. ❌ Skip waypoint information initially (complex)

**Recommended New Observation Shape:**
```python
'aircraft': (max_aircraft, 24)  # Expand from 14 to 24 features
'global_state': (8,)             # Expand from 4 to 8 features
```

---

## 8. ACTION SPACE

### OpenScope Commands

**Available Commands:**
1. **Altitude** - `altitude 10000` or `climb 10000 expedite`
2. **Heading** - `heading 270` or `turn left 45`
3. **Speed** - `speed 250`
4. **Hold** - `hold FIXNAME left 2min`
5. **ILS Approach** - `ils 25L`
6. **Direct to Fix** - `direct FIXNAME`
7. **Cross Fix** - `cross FIXNAME a8000 s250`
8. **SID/STAR** - `climb via SID`, `descend via STAR`
9. **Taxi** - `taxi 25R`
10. **Takeoff** - `takeoff`
11. **Go Around** - `go around`
12. **Cancel Approach** - `cancel approach clearance`
13. **Squawk** - `squawk 1234`
14. **Delete** - `delete` (remove aircraft)

### Self-Contained Environment Actions

**Current Implementation:**
```python
action_space = spaces.Dict({
    'aircraft_id': spaces.Discrete(max_aircraft + 1),  # 0-9 + no-op
    'command_type': spaces.Discrete(5),                # 0-4
    'altitude': spaces.Discrete(18),                   # 0-17
    'heading': spaces.Discrete(12),                    # 0-11
    'speed': spaces.Discrete(8),                       # 0-7
})
```

**Command Types (5):**
- 0: Altitude command
- 1: Heading command
- 2: Speed command
- 3: Land command (assigns runway, begins approach)
- 4: No-op

**Gaps:**
- ❌ No hold command
- ❌ No ILS approach clearance (separate from generic "land")
- ❌ No go-around command
- ❌ No cancel approach command
- ❌ No direct-to-fix (no waypoints)
- ❌ No cross fix with altitude/speed
- ❌ No SID/STAR procedures
- ❌ No ground operations (taxi, takeoff)

**Priority: MEDIUM** - Core commands (alt/hdg/speed/land) cover most scenarios.

### Recommendations:
1. ✅ Add ILS approach command (separate from land)
2. ✅ Add go-around command
3. 🟡 Add cancel approach command
4. ❌ Skip hold/SID/STAR initially (complex)
5. ❌ Skip ground operations (focus on airborne)

**Recommended New Action Space:**
```python
'command_type': spaces.Discrete(7)  # Expand from 5 to 7
# 0: altitude, 1: heading, 2: speed, 3: ILS approach, 4: land, 5: go-around, 6: no-op
```

---

## 9. SPAWNING AND TRAFFIC FLOW

### OpenScope Implementation

**Spawn Mechanisms:**
- **Timed spawns:** Aircraft appear at scheduled times
- **Random spawns:** Configurable spawn rate per airport
- **Spawn locations:**
  - Arrivals: At airspace boundary, on approach course
  - Departures: At gate (APRON phase)
  - Overflights: At airspace boundary, on route
- **Spawn parameters:**
  - Initial altitude (arrivals: cruise, departures: ground)
  - Initial speed (arrivals: cruise speed, departures: 0)
  - Initial heading (arrivals: toward airport, departures: runway heading)
  - Flight plan route
  - Aircraft type (affects performance)

**Traffic Density:**
- Configurable max aircraft count
- Spawn rate adjusts to maintain density
- Balances arrivals and departures

### Self-Contained Environment Implementation

**Current Implementation:**
```python
def _spawn_aircraft(self):
    # Spawn at edge of simulation area
    edge = random.choice([North, East, South, West])

    # Set position at edge
    # Set heading toward center

    altitude = random.uniform(3000, 10000)
    speed = random.uniform(200, 280)

    aircraft = Aircraft(callsign, x, y, altitude, heading, speed)
    self.aircraft.append(aircraft)
```

**Spawning:**
- Every 30 seconds (configurable)
- Random edge position
- Random altitude (3000-10000ft)
- Random speed (200-280kt)
- Heading toward center
- All aircraft are arrivals (no departures)

**Gaps:**
- ❌ No departure spawning
- ❌ No runway-aligned arrival spawning (spawn at edge, not on approach course)
- ❌ No flight plan routes
- ❌ No aircraft type variation
- ❌ No scheduled spawns
- ❌ No overflight aircraft
- ❌ No balance between arrivals and departures

**Priority: MEDIUM** - More realistic spawning improves training diversity.

### Recommendations:
1. ✅ Add departure spawning (at airport, runway assignment)
2. ✅ Spawn arrivals on approach course to runways
3. 🟡 Add arrival/departure ratio control
4. 🟡 Add altitude variation based on distance to airport
5. ❌ Skip flight plan routes initially
6. ❌ Skip aircraft type variation initially
7. ❌ Skip overflight traffic initially

---

## 10. ENVIRONMENTAL FACTORS

### OpenScope Implementation

**Wind:**
- Wind angle (0-360 degrees)
- Wind speed (knots)
- Altitude-dependent wind (increases with altitude)
- Wind vector calculation for physics
- Crosswind/headwind component calculation

**Time of Day:**
- Simulation time tracking
- Day/night cycles (visual only, no gameplay impact)

**Weather:**
- Wind is the primary weather factor
- No precipitation, visibility, or other weather in base game

### Self-Contained Environment Implementation

**Current Implementation:**
- ❌ No wind
- ❌ No time of day
- ❌ No weather effects
- ✅ Time tracking (episode timer)

**Gaps:**
- ❌ No wind simulation
- ❌ No wind effects on aircraft movement
- ❌ No runway preference based on wind
- ❌ No crosswind/tailwind penalties

**Priority: MEDIUM** - Wind adds realism and training complexity.

### Recommendations:
1. ✅ Add constant wind (angle, speed)
2. ✅ Calculate wind effects on groundspeed/track
3. ✅ Calculate crosswind/headwind components per runway
4. ✅ Add wind-based penalties (crosswind, tailwind)
5. 🟡 Add wind variation during episode
6. ❌ Skip time of day (no visual impact in non-browser env)
7. ❌ Skip advanced weather initially

---

## 11. VISUALIZATION AND RENDERING

### OpenScope Implementation

**Canvas Rendering:**
- Radar scope with range rings
- Aircraft icons (triangles) with data tags
- Callsign, altitude, speed labels
- Runway layouts
- Airspace boundaries
- Restricted areas
- Waypoint markers
- ILS centerline visualization
- Conflict/violation visual indicators (red/yellow)
- Storm cells (if enabled)
- Wind indicator

**UI Elements:**
- Command input bar
- Aircraft strip bay
- Score display
- Event log
- Time/traffic count

### Self-Contained Environment Implementation

**Current Implementation:**
```python
def render(self):
    # Black canvas
    # White runways (2 intersecting lines)
    # Yellow triangles for aircraft (rotated by heading)
    # Callsign and altitude labels
    # Info text box (time, aircraft, score, violations, landings)
```

**Gaps:**
- ❌ No range rings
- ❌ No runway labels
- ❌ No ILS centerline visualization
- ❌ No conflict/violation visual indicators
- ❌ No data tags (only callsign + altitude)
- ❌ No speed display
- ❌ No heading indicator
- ❌ No target values display
- ❌ No airspace boundaries
- ❌ No FAF marker
- ❌ No wind indicator

**Priority: LOW** - Visualization is for debugging/demonstration, not RL training.

### Recommendations:
1. ✅ Add FAF markers (circles at 5nm from runway)
2. ✅ Add conflict/violation indicators (red/yellow halos)
3. ✅ Add ILS centerline (extended runway line)
4. 🟡 Add speed to aircraft labels
5. 🟡 Add heading indicator (line extending from triangle)
6. 🟡 Add target altitude display
7. ❌ Skip range rings
8. ❌ Skip airspace boundaries (entire area is controlled)

---

## 12. PERFORMANCE OPTIMIZATION

### OpenScope Implementation

**Optimizations:**
- 8nm bounding box for conflict checks (avoid O(n²) for distant aircraft)
- Quadtree spatial partitioning for aircraft
- Canvas rendering optimizations
- Timewarp support (1x to 10x simulation speed)
- Incremental DOM updates

**Constraints:**
- Browser JavaScript limitations
- Single-threaded execution
- Canvas rendering overhead

### Self-Contained Environment Implementation

**Current Implementation:**
- Pure Python/NumPy
- Simple O(n²) conflict checks
- Matplotlib rendering (slow)
- 1 second per step (fixed)

**Advantages:**
- No browser overhead
- Can use vectorized NumPy operations
- Can use multiprocessing for parallel environments
- Can skip rendering during training

**Gaps:**
- ❌ No spatial partitioning optimization
- ❌ No bounding box pre-filter for conflicts

**Priority: LOW** - Performance is good enough for training.

### Recommendations:
1. 🟡 Add 8nm bounding box pre-filter
2. ❌ Skip quadtree (overkill for <20 aircraft)
3. ✅ Already disabled rendering during training (good!)
4. ✅ Already supports vectorized environments (SB3 DummyVecEnv)

---

## 13. PRIORITY SUMMARY

### Critical Gaps (Implement First) - HIGH PRIORITY

1. **Scoring System** - Add all 14 OpenScope events with correct point values
2. **Flight Phases** - Implement 6-phase state machine (CRUISE, APPROACH, LANDING, etc.)
3. **ILS Approach** - Add FAF, glideslope, course establishment, go-around
4. **Collision Detection** - Separate collision (<0.05nm, -1000pts) from violation
5. **Wind Simulation** - Add constant wind with crosswind/tailwind penalties
6. **Observation Expansion** - Add flight phase, approach clearance, FAF distance, glideslope deviation

### Important Gaps (Implement Second) - MEDIUM PRIORITY

7. **Passing & Diverging** - Add heading-based conflict exemption logic
8. **Command Validation** - 250kt speed limit <10k ft, altitude bounds
9. **Departure Spawning** - Add departures to increase traffic complexity
10. **Acceleration Dynamics** - Add speed acceleration/deceleration (not instant)
11. **Approach Clearance** - Track clearance state, cancel on heading command
12. **Ground Track vs Heading** - Wind drift affects actual path

### Nice-to-Have Gaps (Implement Later) - LOW PRIORITY

13. **Holding Patterns** - Complex but adds ATC realism
14. **SID/STAR Procedures** - Structured routing
15. **Runway Separation** - Category-based spacing enforcement
16. **Visualization Enhancements** - Conflict indicators, ILS centerline, FAF markers
17. **Aircraft Type Variation** - Different performance characteristics
18. **Airspace Boundaries** - Hard boundaries instead of edge spawning

---

## 14. RECOMMENDED IMPLEMENTATION ROADMAP

### Phase 1: Core Game Logic (Week 1)
**Objective:** Match OpenScope scoring and basic flight phases

1. ✅ Implement full scoring system (14 events)
2. ✅ Add collision detection (separate from violations)
3. ✅ Implement 6-phase state machine
4. ✅ Add automatic phase transitions
5. ✅ Expand observation space (flight phase, approach clearance)

**Expected Improvement:** More realistic reward shaping for RL training.

### Phase 2: Approach Procedures (Week 2)
**Objective:** Realistic landing procedures

6. ✅ Add FAF marker (5nm from runway)
7. ✅ Implement ILS approach clearance command
8. ✅ Add course establishment checks (lateral, heading)
9. ✅ Add glideslope altitude calculation (3° path)
10. ✅ Implement automatic go-around logic
11. ✅ Add manual go-around command

**Expected Improvement:** Agent learns proper approach sequencing.

### Phase 3: Wind and Physics (Week 3)
**Objective:** Add environmental realism

12. ✅ Add constant wind (angle, speed)
13. ✅ Calculate wind effects on groundspeed/track
14. ✅ Add crosswind/headwind penalties
15. ✅ Implement acceleration/deceleration dynamics
16. ✅ Add 250kt speed limit below 10,000ft

**Expected Improvement:** Agent learns wind-aware control.

### Phase 4: Traffic Complexity (Week 4)
**Objective:** More realistic ATC scenarios

17. ✅ Add departure aircraft spawning
18. ✅ Spawn arrivals on approach course
19. ✅ Implement passing & diverging logic
20. ✅ Add runway separation enforcement

**Expected Improvement:** Agent handles mixed arrival/departure traffic.

### Phase 5: Polish and Optimization (Week 5)
**Objective:** Production-ready environment

21. ✅ Add visualization enhancements (conflict indicators, ILS, FAF)
22. ✅ Implement 8nm bounding box optimization
23. ✅ Add comprehensive unit tests
24. ✅ Create configuration presets (easy/medium/hard)
25. ✅ Documentation and examples

**Expected Improvement:** Reliable, configurable environment for research.

---

## 15. GAP COVERAGE ESTIMATE

### Current Implementation Coverage

| Category | OpenScope Features | Self-Contained Features | Coverage |
|----------|-------------------|------------------------|----------|
| **Scoring** | 14 events | 5 events | 36% |
| **Flight Phases** | 11 phases | 1 flag | 9% |
| **Collision Logic** | 3 levels | 2 levels | 67% |
| **Commands** | 14 types | 4 types | 29% |
| **Observations** | ~40 features | 14 features | 35% |
| **Approach Procedures** | Full ILS | Simple landing | 10% |
| **Wind Effects** | Full simulation | None | 0% |
| **Traffic** | Arrivals/Departures | Arrivals only | 50% |
| **Physics** | Detailed | Simplified | 60% |
| **Visualization** | Full UI | Basic canvas | 30% |
| **Overall** | - | - | **30%** |

### After Roadmap Completion

| Category | Projected Coverage |
|----------|-------------------|
| **Scoring** | 100% |
| **Flight Phases** | 80% (6 phases vs 11) |
| **Collision Logic** | 100% |
| **Commands** | 50% (7 types vs 14) |
| **Observations** | 70% (~28 features) |
| **Approach Procedures** | 90% |
| **Wind Effects** | 80% |
| **Traffic** | 80% |
| **Physics** | 75% |
| **Visualization** | 60% |
| **Overall** | **75%** |

**Interpretation:** The self-contained environment will capture the essential ATC game logic (scoring, approach procedures, separation rules, wind effects) while omitting complex features not critical for RL training (SID/STAR routing, holding patterns, ground operations).

---

## 16. TESTING STRATEGY

### Unit Tests Required

1. **Scoring System:**
   - Test each of 14 events triggers correct point change
   - Test weighted score calculation
   - Test event counting

2. **Flight Phases:**
   - Test all phase transitions
   - Test automatic transitions (altitude, distance-based)
   - Test phase-specific command restrictions

3. **Collision Detection:**
   - Test collision vs violation vs conflict thresholds
   - Test passing & diverging exemption logic
   - Test low-altitude exemption

4. **Approach Procedures:**
   - Test course establishment (lateral, heading)
   - Test glideslope calculation at various distances
   - Test automatic go-around trigger
   - Test manual go-around

5. **Wind Simulation:**
   - Test wind vector calculation
   - Test groundspeed/track calculation
   - Test crosswind/headwind component calculation
   - Test wind penalties

6. **Command Validation:**
   - Test 250kt speed limit enforcement
   - Test altitude bounds
   - Test approach clearance cancellation

### Integration Tests Required

1. **Complete Landing Sequence:**
   - Spawn aircraft on approach
   - Issue ILS clearance
   - Verify APPROACH phase transition
   - Verify LANDING phase transition at FAF
   - Verify successful landing and +10 score

2. **Separation Violation:**
   - Spawn two aircraft
   - Command them to converging courses
   - Verify conflict detection
   - Verify violation penalty
   - Verify no penalty if diverging

3. **Go-Around Scenario:**
   - Aircraft on approach
   - Aircraft deviates from glideslope
   - Verify automatic go-around
   - Verify -50 penalty
   - Verify return to DESCENT phase

4. **Wind Penalty:**
   - Set strong tailwind
   - Aircraft lands
   - Verify tailwind penalty applied
   - Set strong crosswind
   - Verify crosswind penalty applied

### Benchmark Tests

1. **Performance:** 100 aircraft for 1000 steps < 10 seconds
2. **Determinism:** Same seed → same episode
3. **Gymnasium Compliance:** Pass `gym.utils.env_checker.check_env()`

---

## 17. CONFIGURATION PRESETS

### Easy Mode (For Initial Training)
```yaml
max_aircraft: 5
spawn_interval: 40.0  # seconds
episode_length: 300   # 5 minutes
wind_speed: 0         # No wind
enable_departures: false
separation_lateral_nm: 5.0  # Relaxed separation
enable_go_around: false
enable_wind_penalties: false
```

### Medium Mode (Balanced Training)
```yaml
max_aircraft: 10
spawn_interval: 25.0
episode_length: 600   # 10 minutes
wind_speed: 10        # Moderate wind
enable_departures: true
departure_ratio: 0.3  # 30% departures
separation_lateral_nm: 3.0  # Standard separation
enable_go_around: true
enable_wind_penalties: true
```

### Hard Mode (Full Realism)
```yaml
max_aircraft: 20
spawn_interval: 15.0
episode_length: 900   # 15 minutes
wind_speed: 20        # Strong wind
enable_departures: true
departure_ratio: 0.5  # 50% departures
separation_lateral_nm: 3.0
enable_go_around: true
enable_wind_penalties: true
enable_passing_diverging: true
enable_runway_separation: true
```

---

## 18. CONCLUSION

The self-contained Python environment currently implements **~30% of OpenScope's game logic**, covering basic aircraft movement, simple separation rules, and primitive landing detection. To create an effective RL training environment that captures the essence of ATC operations, the following high-priority additions are recommended:

**Critical Additions:**
1. ✅ **Full scoring system** (14 events) - Essential for reward shaping
2. ✅ **Flight phase state machine** - Core to ATC workflow
3. ✅ **ILS approach procedures** - Central to landing sequence
4. ✅ **Wind simulation** - Adds environmental realism
5. ✅ **Collision detection** - Severe penalty differentiation
6. ✅ **Observation expansion** - Richer state information for policy

**Important Additions:**
7. ✅ **Passing & diverging logic** - Realistic separation rules
8. ✅ **Command validation** - Enforces constraints
9. ✅ **Departure traffic** - Increases scenario complexity
10. ✅ **Acceleration dynamics** - More realistic physics

With these additions, the environment will reach **~75% coverage** of OpenScope's core mechanics while maintaining the advantages of a self-contained Python implementation: no browser dependencies, faster execution, easier integration with RL frameworks, and better reproducibility.

The proposed 5-week roadmap provides a structured path to building a production-ready ATC training environment suitable for reinforcement learning research.

---

## APPENDIX: Key OpenScope Source Files

| Feature | File Path |
|---------|-----------|
| Scoring | `/openscope/src/assets/scripts/client/game/GameController.js` |
| Flight Phases | `/openscope/src/assets/scripts/client/constants/aircraftConstants.js` |
| Aircraft Model | `/openscope/src/assets/scripts/client/aircraft/AircraftModel.js` |
| Collision Detection | `/openscope/src/assets/scripts/client/aircraft/AircraftConflict.js` |
| Commands | `/openscope/src/assets/scripts/client/commands/aircraftCommand/` |
| ILS/Approach | `/openscope/src/assets/scripts/client/airport/runway/RunwayModel.js` |
| Pilot Logic | `/openscope/src/assets/scripts/client/aircraft/Pilot/Pilot.js` |
| Wind Calculation | `/openscope/src/assets/scripts/client/airport/AirportModel.js` |
| Performance Constants | `/openscope/src/assets/scripts/client/constants/aircraftConstants.js` |

---

**End of Gap Analysis**
