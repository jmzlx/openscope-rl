# OpenScope RL Test Suite

This directory contains comprehensive tests for the OpenScope RL project.

## Test Structure

### `test_openscope_integration.py`
Tests for OpenScope integration:
- Browser management (initialization, navigation, cleanup)
- Game interface (command execution, state extraction)
- Full environment integration (requires OpenScope server)
- **Event detection and scoring validation** - Verifies that all scoring events are captured and score calculation is accurate

### `test_data_extraction.py`
Tests for data extraction and processing:
- State extraction from OpenScope
- State processing into observations
- Data normalization
- Handling of missing data
- Data completeness validation

### `test_model_data_requirements.py`
Tests for model data requirements:
- Observation compatibility with models
- Data type and shape validation
- End-to-end data flow (state → observation → model input)
- Batch processing
- Model validation

### `test_interface_abstraction.py`
Tests for interface abstraction layer:
- MockATCEnvironment implementation
- OpenScopeAdapter wrapping
- Model compatibility with interface abstraction
- Separation of concerns

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run specific test file:
```bash
pytest tests/test_data_extraction.py
```

### Run with verbose output:
```bash
pytest tests/ -v
```

### Run specific test class:
```bash
pytest tests/test_data_extraction.py::TestStateProcessing
```

### Run with coverage:
```bash
pytest tests/ --cov=environment --cov=models
```

## Test Categories

### Unit Tests
- Test individual components in isolation
- Use mocks where appropriate
- Fast execution
- No external dependencies

### Integration Tests
- Test component interactions
- Some tests require OpenScope server (marked with `@pytest.mark.skip`)
- May require browser installation

### Event Detection Test
The `test_event_detection_and_scoring` test is critical for RL training accuracy:
- **Purpose**: Validates that all OpenScope scoring events are captured
- **Validation**: Compares calculated score from events against actual game score
- **Requirements**: OpenScope server running at localhost:3003
- **Duration**: ~25 seconds (includes 20s high-speed simulation)
- **Events tested**: All 14 OpenScope event types (ARRIVAL, DEPARTURE, COLLISION, AIRSPACE_BUST, etc.)

To run this test:
```bash
# Start OpenScope server first
cd ../openscope && npm run start

# Run the event detection test
pytest tests/test_openscope_integration.py::TestOpenScopeIntegration::test_event_detection_and_scoring -v -s
```

The test will output:
- Initial and final scores
- All captured event types and counts
- Calculated vs actual score comparison
- Pass/fail status with detailed breakdown

### Manual Tests
Tests marked with `@pytest.mark.skip(reason="Requires OpenScope server")` should be run manually:
1. Start OpenScope server: `cd ../openscope && npm run start`
2. Run the test manually or remove the skip decorator

## Fixtures

Common fixtures are defined in `conftest.py`:
- `mock_game_state`: Mock game state data
- `mock_observation`: Mock observation dictionary
- `mock_action`: Mock action dictionary
- `test_config`: Test configuration
- `default_config`: Default OpenScopeConfig
- `mock_page`: Mock Playwright page
- `observation_space`: Observation space
- `action_space`: Action space
- `mock_model_input`: Model input format

## Writing New Tests

### Example Unit Test:
```python
def test_feature_extraction(mock_game_state):
    """Test feature extraction from game state."""
    processor = StateProcessor(default_config)
    obs = processor.process_state(mock_game_state)
    
    assert "aircraft" in obs
    assert obs["aircraft"].shape[0] == default_config.max_aircraft
```

### Example Integration Test:
```python
@pytest.mark.skip(reason="Requires OpenScope server - run manually")
def test_full_integration():
    """Test full integration with OpenScope."""
    env = PlaywrightEnv(airport="KLAS", max_aircraft=5)
    try:
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert "aircraft" in obs
    finally:
        env.close()
```

## Continuous Integration

Tests should pass in CI/CD:
- All unit tests (no external dependencies)
- Fast execution (< 1 minute for unit tests)
- Integration tests can be run manually or on schedule

## Test Coverage Goals

- **Unit tests**: > 80% coverage
- **Integration tests**: Critical paths covered
- **Data extraction**: 100% coverage of required fields
- **Model compatibility**: All model inputs validated

## Notes

- Tests use mocks extensively to avoid requiring OpenScope server
- Integration tests are marked and should be run manually
- Mock implementations are provided in `environment/interface.py` for testing models
- All tests should be deterministic (use fixed seeds where needed)

