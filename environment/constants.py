"""
Constants module for OpenScope RL environment.

This module defines all constants used throughout the environment,
including observation dimensions, action mappings, and game parameters.
"""

from enum import Enum
from typing import List


class CommandType(Enum):
    """Available command types for aircraft."""
    ALTITUDE = "altitude"
    HEADING = "heading"
    SPEED = "speed"
    ILS = "ils"
    DIRECT = "direct"


class AircraftCategory(Enum):
    """Aircraft categories."""
    ARRIVAL = "arrival"
    DEPARTURE = "departure"


class GameState(Enum):
    """Game state indicators."""
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


# Observation space dimensions
AIRCRAFT_FEATURE_DIM = 14
GLOBAL_STATE_DIM = 4
MAX_AIRCRAFT_DEFAULT = 20

# Action space dimensions
COMMAND_TYPE_COUNT = 5
ALTITUDE_LEVELS = 18
HEADING_CHANGES_COUNT = 13
SPEED_LEVELS = 8

# Action mappings
ALTITUDE_VALUES: List[int] = [
    10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180
]

HEADING_CHANGES: List[int] = [
    -90, -60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60, 90
]

SPEED_VALUES: List[int] = [
    180, 200, 220, 240, 260, 280, 300, 320
]

COMMAND_TYPES: List[CommandType] = [
    CommandType.ALTITUDE,
    CommandType.HEADING,
    CommandType.SPEED,
    CommandType.ILS,
    CommandType.DIRECT
]

# Game parameters
DEFAULT_GAME_URL = "http://localhost:3003"
DEFAULT_AIRPORT = "KLAS"
DEFAULT_TIMEWARP = 5
DEFAULT_EPISODE_LENGTH = 3600
DEFAULT_ACTION_INTERVAL = 5.0

# Browser settings
DEFAULT_BROWSER_ARGS = [
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--disable-gpu',
    '--disable-web-security',
    '--disable-features=VizDisplayCompositor',
    '--memory-pressure-off',
    '--max_old_space_size=512'
]

# Timing constants
PAGE_LOAD_TIMEOUT = 2.0
AIRPORT_SETUP_DELAY = 2.0  # Reduced from 10.0 - game loads much faster
TIMEWARP_SETUP_DELAY = 0.3  # Reduced from 0.5
GAME_UPDATE_DELAY = 0.02

# Normalization constants
POSITION_SCALE_FACTOR = 100.0
MAX_ANGLE = 360.0
MAX_SPEED = 500.0
MAX_ALTITUDE = 50000.0
MAX_GROUND_SPEED = 600.0

# Episode termination conditions
MIN_SCORE_THRESHOLD = -2000
MAX_STEPS_THRESHOLD = 1000

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# JavaScript functions for OpenScope game interaction
JS_EXECUTE_COMMAND_SCRIPT = """
(command) => {
    const input = document.querySelector('#command');
    if (input) {
        input.value = command;
        input.dispatchEvent(new KeyboardEvent('keydown', {
            key: 'Enter', code: 'Enter', keyCode: 13
        }));
    }
}
"""

JS_GET_GAME_STATE_SCRIPT = """
(function() {
    if (!window.aircraftController) return null;

    const aircraft = [];
    for (const ac of window.aircraftController.aircraft.list) {
        aircraft.push({
            callsign: ac.callsign,
            position: ac.relativePosition,
            altitude: ac.altitude,
            heading: ac.heading,
            speed: ac.speed,
            groundSpeed: ac.groundSpeed,
            assignedAltitude: ac.mcp.altitude,
            assignedHeading: ac.mcp.heading,
            assignedSpeed: ac.mcp.speed,
            category: ac.category,
            isOnGround: ac.isOnGround(),
            isTaxiing: ac.isTaxiing(),
            isEstablished: ac.isEstablishedOnCourse ? ac.isEstablishedOnCourse() : false,
            targetRunway: ac.fms && ac.fms.arrivalRunwayModel ? ac.fms.arrivalRunwayModel.name : null,
        });
    }

    const conflicts = [];
    if (window.aircraftController.conflicts) {
        window.aircraftController.conflicts.forEach(c => {
            conflicts.push({
                aircraft1: c.aircraft[0].callsign,
                aircraft2: c.aircraft[1].callsign,
                distance: c.distance,
                altitude: c.altitude,
                hasConflict: c.hasConflict(),
                hasViolation: c.hasViolation()
            });
        });
    }

    // Get game time
    let gameTime = 0;
    if (window._getRLGameTime) {
        try { gameTime = window._getRLGameTime(); } catch (e) {}
    }

    // Get score from DOM (OpenScope doesn't expose gameController anymore)
    const scoreElement = document.querySelector('#score');
    const score = scoreElement ? (parseInt(scoreElement.textContent) || 0) : 0;

    return {
        aircraft: aircraft,
        conflicts: conflicts,
        score: score,
        time: gameTime,
        numAircraft: aircraft.length
    };
})();
"""

JS_GET_ENHANCED_GAME_STATE_SCRIPT = """
(function() {
    if (!window.aircraftController) return null;

    const aircraft = [];
    for (const ac of window.aircraftController.aircraft.list) {
        const aircraftData = {
            // Basic properties
            callsign: ac.callsign,
            position: ac.relativePosition,
            altitude: ac.altitude,
            heading: ac.heading,
            speed: ac.speed,
            groundSpeed: ac.groundSpeed,
            assignedAltitude: ac.mcp.altitude,
            assignedHeading: ac.mcp.heading,
            assignedSpeed: ac.mcp.speed,
            category: ac.category,
            isOnGround: ac.isOnGround(),
            isTaxiing: ac.isTaxiing(),
            
            // Additional properties
            isEstablished: ac.isEstablishedOnCourse ? ac.isEstablishedOnCourse() : false,
            targetRunway: ac.fms.arrivalRunwayModel ? ac.fms.arrivalRunwayModel.name : null,
            
            // Try to get additional properties if they exist
            verticalSpeed: ac.verticalSpeed || null,
            flightPhase: ac.flightPhase || null,
            squawk: ac.squawk || null,
            pitch: ac.pitch || null,
            roll: ac.roll || null,
            yaw: ac.yaw || null,
            mach: ac.mach || null,
            trueAirspeed: ac.trueAirspeed || null,
            calibratedAirspeed: ac.calibratedAirspeed || null,
            groundTrack: ac.groundTrack || null,
            course: ac.course || null,
            track: ac.track || null,
            bearing: ac.bearing || null,
            
            // Wind-related properties (if available)
            windSpeed: ac.windSpeed || null,
            windDirection: ac.windDirection || null,
            headwind: ac.headwind || null,
            tailwind: ac.tailwind || null,
            crosswind: ac.crosswind || null,
            
            // Position properties
            absolutePosition: ac.absolutePosition || null,
            lat: ac.lat || null,
            lon: ac.lon || null,
            latitude: ac.latitude || null,
            longitude: ac.longitude || null,
            
            // Additional useful properties
            distanceToRunway: ac.distanceToRunway || null,
            timeToRunway: ac.timeToRunway || null,
            eta: ac.eta || null,
            fuel: ac.fuel || null,
            weight: ac.weight || null,
            mass: ac.mass || null,
            
            // FMS/Flight plan data
            flightPlan: ac.fms ? {
                waypoints: ac.fms.waypoints ? ac.fms.waypoints.map(wp => ({
                    name: wp.name,
                    position: wp.position,
                    altitude: wp.altitude
                })) : null,
                currentWaypoint: ac.fms.currentWaypoint ? ac.fms.currentWaypoint.name : null,
                nextWaypoint: ac.fms.nextWaypoint ? ac.fms.nextWaypoint.name : null
            } : null
        };
        
        aircraft.push(aircraftData);
    }

    const conflicts = [];
    if (window.aircraftController.conflicts) {
        window.aircraftController.conflicts.forEach(c => {
            conflicts.push({
                aircraft1: c.aircraft[0].callsign,
                aircraft2: c.aircraft[1].callsign,
                distance: c.distance,
                altitude: c.altitude,
                hasConflict: c.hasConflict(),
                hasViolation: c.hasViolation()
            });
        });
    }

    // Get game time
    let gameTime = 0;
    if (window._getRLGameTime) {
        try { gameTime = window._getRLGameTime(); } catch (e) {}
    }

    // Get score from DOM (OpenScope doesn't expose gameController anymore)
    const scoreElement = document.querySelector('#score');
    const score = scoreElement ? (parseInt(scoreElement.textContent) || 0) : 0;

    // Note: Weather data may not be available without gameController
    // This would need to be accessed through DOM or other means if needed
    const weather = {};

    return {
        aircraft: aircraft,
        conflicts: conflicts,
        score: score,
        time: gameTime,
        weather: weather,
        numAircraft: aircraft.length
    };
})();
"""

# Optimal extraction script - extracts the 14 required features per aircraft plus
# additional ATC-critical data for maximum efficiency in production training data collection
JS_GET_OPTIMAL_GAME_STATE_SCRIPT = """
(function() {
    if (!window.aircraftController) return null;

    const aircraft = [];
    for (const ac of window.aircraftController.aircraft.list) {
        // Extract wind components
        let windComponents = {cross: 0, head: 0};
        try {
            if (ac.getWindComponents && typeof ac.getWindComponents === 'function') {
                windComponents = ac.getWindComponents();
            }
        } catch (e) {
            windComponents = {cross: 0, head: 0};
        }
        
        // Extract FMS/waypoint data safely
        let nextWaypoint = null;
        let currentWaypoint = null;
        let flightPlanAltitude = null;
        let flightPlanRoute = null;
        let targetRunway = null;
        try {
            if (ac.fms) {
                nextWaypoint = (ac.fms.nextWaypoint && ac.fms.nextWaypoint.name) ? ac.fms.nextWaypoint.name : null;
                currentWaypoint = (ac.fms.currentWaypoint && ac.fms.currentWaypoint.name) ? ac.fms.currentWaypoint.name : null;
                flightPlanAltitude = (ac.fms.flightPlanAltitude !== undefined && ac.fms.flightPlanAltitude !== null) ? ac.fms.flightPlanAltitude : null;
                if (ac.fms.getFullRouteStringWithoutAirportsWithSpaces && typeof ac.fms.getFullRouteStringWithoutAirportsWithSpaces === 'function') {
                    flightPlanRoute = ac.fms.getFullRouteStringWithoutAirportsWithSpaces();
                }
                // Extract target runway (critical for ILS commands and action masking)
                targetRunway = (ac.fms.arrivalRunwayModel && ac.fms.arrivalRunwayModel.name) ? ac.fms.arrivalRunwayModel.name : null;
            }
        } catch (e) {
            // FMS data extraction failed, leave as null
        }
        
        // Extract pilot/approach data safely
        let hasApproachClearance = false;
        try {
            if (ac.pilot && ac.pilot.hasApproachClearance !== undefined) {
                hasApproachClearance = ac.pilot.hasApproachClearance;
            }
        } catch (e) {
            hasApproachClearance = false;
        }
        
        // Extract landing status safely
        let isOnFinal = false;
        let isEstablishedOnGlidepath = false;
        try {
            if (ac.isOnFinal && typeof ac.isOnFinal === 'function') {
                isOnFinal = ac.isOnFinal();
            }
            if (ac.isEstablishedOnGlidepath && typeof ac.isEstablishedOnGlidepath === 'function') {
                isEstablishedOnGlidepath = ac.isEstablishedOnGlidepath();
            }
        } catch (e) {
            // Methods not available or failed
        }
        
        // Extract climb rate/vertical speed safely
        let climbRate = null;
        try {
            if (ac.getClimbRate && typeof ac.getClimbRate === 'function') {
                climbRate = ac.getClimbRate();
            }
        } catch (e) {
            climbRate = null;
        }
        
        aircraft.push({
            callsign: ac.callsign,
            position: ac.relativePosition,                    // Features 0, 1: x, y position
            altitude: ac.altitude,                             // Feature 2: altitude
            heading: ac.heading,                               // Feature 3: heading
            speed: ac.speed,                                    // Feature 4: speed
            groundSpeed: ac.groundSpeed,                       // Feature 5: ground speed
            assignedAltitude: ac.mcp.altitude,                 // Feature 6: assigned altitude
            assignedHeading: ac.mcp.heading,                   // Feature 7: assigned heading
            assignedSpeed: ac.mcp.speed,                       // Feature 8: assigned speed
            isOnGround: ac.isOnGround(),                       // Feature 9: is on ground
            isTaxiing: ac.isTaxiing(),                         // Feature 10: is taxiing
            isEstablished: ac.isEstablishedOnCourse ? ac.isEstablishedOnCourse() : false, // Feature 11: is established
            category: ac.category,                             // Features 12, 13: is_arrival, is_departure
            
            // Additional ATC-critical data
            windComponents: windComponents,                     // Wind: crosswind and headwind components
            flightPhase: ac.flightPhase || null,               // Current flight phase
            nextWaypoint: nextWaypoint,                         // Next waypoint name
            currentWaypoint: currentWaypoint,                   // Current waypoint name
            flightPlanAltitude: flightPlanAltitude,            // Flight plan altitude
            flightPlanRoute: flightPlanRoute,                  // Flight plan route string
            targetRunway: targetRunway,                        // Target runway (critical for ILS commands)
            hasApproachClearance: hasApproachClearance,        // Approach clearance status
            isOnFinal: isOnFinal,                              // Is on final approach
            isEstablishedOnGlidepath: isEstablishedOnGlidepath, // Is established on glidepath
            
            // Operational state fields
            isControllable: ac.isControllable !== undefined ? ac.isControllable : true, // Whether aircraft can receive commands
            transponderCode: ac.transponderCode || null,       // Transponder/squawk code
            groundTrack: ac.groundTrack !== undefined ? ac.groundTrack : null, // Actual track over ground (radians)
            trueAirspeed: ac.trueAirspeed !== undefined ? ac.trueAirspeed : null, // True airspeed vs indicated
            climbRate: climbRate,                              // Vertical speed/climb rate (ft/min)
            distance: ac.distance !== undefined ? ac.distance : null, // Distance to airport/reference (nm)
        });
    }

    const conflicts = [];
    if (window.aircraftController.conflicts) {
        window.aircraftController.conflicts.forEach(c => {
            conflicts.push({
                aircraft1: c.aircraft[0].callsign,
                aircraft2: c.aircraft[1].callsign,
                distance: c.distance,
                altitude: c.altitude,
                hasConflict: c.hasConflict(),
                hasViolation: c.hasViolation()
            });
        });
    }

    // Get game time
    let gameTime = 0;
    if (window._getRLGameTime) {
        try { gameTime = window._getRLGameTime(); } catch (e) {}
    }

    // Get score from DOM
    const scoreElement = document.querySelector('#score');
    const score = scoreElement ? (parseInt(scoreElement.textContent) || 0) : 0;

    return {
        aircraft: aircraft,
        conflicts: conflicts,
        score: score,
        time: gameTime,
        numAircraft: aircraft.length
    };
})();
"""
