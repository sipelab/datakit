# DataKit Data Sources

This module contains all data source implementations for loading various types of experimental data. Data sources are organized by category and support versioning for backward compatibility and feature evolution.

## Organization

The data sources are organized into the following categories:

### Camera Data Sources (`camera/`)
- **MesoMetadataSource** (`mesoscope.py`) - Loads mesoscope camera metadata JSON files
- **PupilMetadataSource** (`pupil.py`) - Loads pupil camera metadata JSON files

### Behavioral Data Sources (`behavior/`)
- **TreadmillSource** (`treadmill.py`) - Loads treadmill speed and distance data with advanced alignment
- **DataqueueSource** (`dataqueue.py`) - Loads dataqueue synchronization data

### Analysis Data Sources (`analysis/`)
- **MesoMeanSource** (`mesoscope.py`) - Loads mesoscope mean fluorescence traces
- **PupilDLCSource** (`pupil.py`) - Loads DeepLabCut pupil analysis results

### Session Data Sources (`session/`)
- **SessionConfigSource** (`config.py`) - Loads session configuration parameters
- **SessionNotesSource** (`notes.py`) - Loads timestamped session notes

## Versioning System

Data sources support versioning to enable:
- Backward compatibility when changing implementations
- A/B testing of different processing approaches
- Gradual migration to improved algorithms

### Version Examples

```python
from datakit.loaders import DataSource

# Load latest version (default)
treadmill = DataSource.create_loader("treadmill")

# Load specific version
treadmill_v1 = DataSource.create_loader("treadmill", "1.0")  # Simple CSV loading
treadmill_v2 = DataSource.create_loader("treadmill", "2.0")  # Advanced alignment

# Check available versions
versions = DataSource.get_available_versions("treadmill")
print(versions)  # ['1.0', '2.0']
```

### Version History

#### TreadmillSource
- **v1.0**: Simple CSV loading with basic timestamp conversion
- **v2.0**: Advanced alignment with dataqueue synchronization, 32-bit wraparound correction, and locomotion bout detection

#### Other Sources
- **v1.0**: Initial implementations (all other sources currently at v1.0)

## Adding New Data Sources

#TODO better clarification here for dev interest
You will need to define a subclassed `DataSource` with several key class attributes. You won't need an __init__ as all DataSource subclasses are registered whenever datakit is imported. Sweet deal. The `load` method needs to return a `LoadedStream`. If you can load and shape your data to fit the requirements of a `LoadedStream` then you're ready to cook. Tip: use class attributes to parameterize your class--they can be made mutable or not (#TODO how?)

To add a new data source:

1. Choose the appropriate category directory
2. Create a new Python file
3. Implement a class inheriting from `DataSource`
4. Set the required class variables:
   - `tag`: Unique identifier (e.g., "new_sensor")
   - `patterns`: File patterns to match (e.g., `("**/*_sensor.csv",)`)
   - `camera_tag`: Camera binding if applicable (or `None`)
   - `version`: Version string (start with "1.0")
5. Implement the `load()` method returning a `LoadedStream`
6. Add import to `__init__.py`

### Example New Data Source

```python
from datakit.loaders import DataSource
from datakit.datamodel import LoadedStream

class NewSensorSource(DataSource):
    """Load new sensor data."""
    tag = "new_sensor"
    patterns = ("**/*_sensor.csv",)
    camera_tag = None
    version = "1.0"
    
    def load(self, path: Path) -> LoadedStream:
        # Implementation here
        pass
```

