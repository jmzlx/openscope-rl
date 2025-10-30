# Testing Strategy and Architecture

This document describes the testing strategy and architecture improvements made to ensure we don't regress functionality.

## Overview

We've created a comprehensive test suite and refactored the codebase to separate OpenScope interaction from models through an interface abstraction layer.

## Test Suite

### Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and test utilities
├── test_openscope_integration.py   # OpenScope integration tests
├── test_data_extraction.py          # Data extraction tests
├── test_model_data_requirements.py  # Model data validation tests
├── test_interface_abstraction.py   # Interface abstraction tests
└── README.md                       # Test documentation
```

### Test Categories

1. **OpenScope Integration Tests** (`test_openscope_integration.py`)
   - Browser management (initialization, navigation, cleanup)
   - Game interface (command execution, state extraction)
   - Environment initialization and reset
   - Full integration (requires OpenScope server - marked as skipped)

2. **Data Extraction Tests** (`test_data_extraction.py`)
   - State extraction structure and completeness
   - State processing into observations
   - Data normalization correctness
   - Handling of missing/invalid data
   - Data completeness validation

3. **Model Data Requirements Tests** (`test_model_data_requirements.py`)
   - Observation compatibility with models
   - Data type and shape validation
   - End-to-end data flow (state → observation → model input)
   - Batch processing support
   - Model input validation

4. **Interface Abstraction Tests** (`test_interface_abstraction.py`)
   - MockATCEnvironment implementation
   - OpenScopeAdapter wrapping
   - Model compatibility with interface

## Interface Abstraction Layer

### Architecture

To separate OpenScope interaction from models, we've created an abstraction layer in `environment/interface.py`:

```
┌─────────────────────────────────────────┐
│         Models Module                   │
│  (Models only depend on interface)     │
└──────────────┬──────────────────────────┘
               │
               │ depends on
               ▼
┌─────────────────────────────────────────┐
│    ATCEnvironmentInterface              │
│    (Abstract interface)                  │
└──────────────┬──────────────────────────┘
               │
      ┌────────┴────────┐
      │                  │
      ▼                  ▼
┌─────────────┐  ┌──────────────┐
│OpenScope    │  │MockATC       │
│Adapter      │  │Environment   │
└──────┬──────┘  └──────────────┘
       │
       │ wraps
       ▼
┌─────────────────────────────────────────┐
│    PlaywrightEnv                        │
│    (OpenScope implementation)          │
└─────────────────────────────────────────┘
```

### Benefits

1. **Separation of Concerns**: Models don't depend on OpenScope-specific code
2. **Testability**: Easy to test models with mock environments
3. **Flexibility**: Can swap OpenScope with other simulators
4. **Clear Contract**: Interface defines exactly what models need

### Components

- **`ATCEnvironmentInterface`**: Abstract base class defining the contract
- **`OpenScopeAdapter`**: Wraps `PlaywrightEnv` to implement the interface
- **`MockATCEnvironment`**: Mock implementation for testing models
- **`ATCStateExtractor`**: Abstract interface for state extraction
- **`ATCCommandExecutor`**: Abstract interface for command execution

## Running Tests

### All Tests
```bash
pytest tests/
```

### Specific Categories
```bash
# Integration tests
pytest tests/test_openscope_integration.py

# Data extraction tests
pytest tests/test_data_extraction.py

# Model compatibility tests
pytest tests/test_model_data_requirements.py

# Interface abstraction tests
pytest tests/test_interface_abstraction.py
```

### With Coverage
```bash
pytest tests/ --cov=environment --cov=models --cov-report=html
```

## Test Fixtures

Common fixtures in `conftest.py`:
- `mock_game_state`: Sample game state data
- `mock_observation`: Sample observation dictionary
- `mock_action`: Sample action dictionary
- `default_config`: Default configuration
- `mock_page`: Mock Playwright page for testing

## What Gets Tested

### OpenScope Integration
✅ Browser initialization and cleanup
✅ Game interface initialization
✅ Command execution
✅ State extraction
✅ Game readiness checks
✅ Environment reset and step

### Data Extraction
✅ State extraction structure
✅ All required aircraft fields present
✅ State processing produces valid observations
✅ Normalization is correct
✅ Missing data is handled gracefully
✅ Observation validation works

### Model Requirements
✅ Observations compatible with model inputs
✅ All required data fields present
✅ Data types match expectations
✅ Shapes match model expectations
✅ No NaN or Inf values
✅ End-to-end pipeline works
✅ Batch processing works

### Interface Abstraction
✅ Mock environment implements interface
✅ Adapter wraps environment correctly
✅ Models work with any interface implementation
✅ Separation of concerns is maintained

## Continuous Integration

Tests should:
- Run quickly (< 1 minute for unit tests)
- Be deterministic (use fixed seeds)
- Not require external services (use mocks)
- Integration tests can be run manually or on schedule

## Integration Tests

Some tests require OpenScope server running:
- Marked with `@pytest.mark.skip(reason="Requires OpenScope server")`
- To run manually:
  1. Start OpenScope: `cd ../openscope && npm run start`
  2. Run test or remove skip decorator

## Regression Prevention

The test suite prevents regression by:
1. **Testing critical paths**: Integration, data extraction, model compatibility
2. **Validating data contracts**: Ensures observations match model expectations
3. **Interface abstraction**: Models work with any implementation
4. **Comprehensive coverage**: All major components tested

## Future Improvements

- Add property-based testing for data validation
- Add performance benchmarks
- Add visualization tests
- Add more integration scenarios
- Add tests for all training scripts

