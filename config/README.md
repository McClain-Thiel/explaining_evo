# Configuration System

This project uses a Pydantic-based configuration system that allows you to manage settings through config files, environment variables, and command-line arguments.

## Using Config Files

Config files can be in YAML or JSON format. To use a config file, simply specify it with the `--config` flag:

```bash
python -m src.explaining.run_evo2_analysis --config config/evo2_analysis.yaml
python -m src.explaining.scripts.query_database --config config/database_query.yaml list-feature-sets
python -m src.explaining.scripts.analyze_features --config config/feature_analysis.yaml visualize-features --feature_set_id 1
```

## Configuration Priority

Settings are loaded in the following order of priority (highest to lowest):

1. Command-line arguments
2. Environment variables
3. Config file
4. Default values

This means you can override any setting from the config file by specifying it as a command-line argument.

## Environment Variables

You can also use environment variables to configure the application. Environment variables should be prefixed according to their section:

- `DB_` for database settings
- `MODEL_` for model settings
- `AUTOENCODER_` for autoencoder settings
- `PROCESSING_` for processing settings
- `QUERY_` for query settings
- `FEATURE_ANALYSIS_` for feature analysis settings

For example:

```bash
# Set database URL via environment variable
export DB_URL="postgresql://user:password@localhost:5432/my_database"

# Run with environment variable settings
python -m src.explaining.scripts.query_database list-feature-sets
```

## Sample Configuration Files

The `config/` directory contains several sample configuration files:

- `evo2_analysis.yaml`: Configuration for running the full Evo2 analysis pipeline
- `database_query.yaml`: Configuration for database query operations
- `feature_analysis.yaml`: Configuration for feature analysis operations

## Creating Custom Configurations

You can create your own configuration files by copying and modifying the sample files. You only need to include the settings you want to override; default values will be used for any settings not specified.

## Validation

All configuration settings are validated to ensure they contain valid values. If a setting has an invalid value, an error will be displayed explaining the problem. 