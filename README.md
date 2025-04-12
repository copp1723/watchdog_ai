# Watchdog AI

Watchdog AI is a dealership data normalization and analysis tool designed to help automotive dealerships standardize, validate, and analyze their data across various systems (CDK, Reynolds, DealerTrack, etc.).

## Features

- **Template Detection**: Automatically identify data formats from common DMS systems
- **Field Standardization**: Normalize field names across different data sources
- **Data Validation**: Check for data quality issues and provide detailed reports
- **AI-Powered Analysis**: Get insights and recommendations using Claude's AI capabilities
- **Interactive UI**: Easy-to-use Streamlit interface for uploading and analyzing data
- **PDF Parsing**: Extract data from PDF reports (configurable)

## Getting Started

### Prerequisites

- Python 3.9+
- Required packages (see `requirements.txt`)
- Anthropic API key for Claude integration

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/copp1723/watchdog_ai.git
   cd watchdog_ai
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your Anthropic API key
   ```

### Running the Application

Run the Streamlit application:
```bash
streamlit run app.py
```

This will start the web application, which you can access at http://localhost:8501.

## Project Structure

```
watchdog_ai/
├── .env.example                # Template for environment variables
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── app.py                      # Main Streamlit application
├── assets/
│   ├── data_dictionary.json    # Data dictionary for field mapping
│   └── sample_data/            # Sample datasets
├── config/
│   └── config.py               # Configuration settings
├── src/
│   ├── data/
│   │   ├── template_mapper.py  # Field mapping and standardization
│   │   ├── parsers/            # File parsing modules
│   │   └── validators/         # Data validation
│   ├── ai/
│   │   ├── claude_client.py    # Claude API client
│   │   └── prompts.py          # Prompts for AI analysis
│   ├── ui/
│   │   └── pages/              # Streamlit UI pages
│   └── services/
│       ├── storage.py          # Data storage service (future)
│       └── digest.py           # Report generation (future)
└── tests/
    ├── test_template_mapper.py # Unit tests
    └── test_data/              # Test datasets
```

## Using the Application

1. **Upload Data**: Upload your dealership data files (CSV, Excel, PDF)
2. **Review Mapping**: Verify the automatic field mapping and make adjustments
3. **Validate Data**: Check for data quality issues and understand potential problems
4. **Analyze Results**: Review insights and recommendations provided by the AI
5. **Export Data**: Download processed data in standardized format

## Data Dictionary

The system uses a data dictionary (`assets/data_dictionary.json`) to define:
- Field mappings between original and standardized names
- Document templates for known DMS systems
- Validation rules and requirements

You can customize this dictionary to add support for additional fields or templates.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Anthropic Claude](https://www.anthropic.com/claude) for AI capabilities
- [Streamlit](https://streamlit.io/) for the web interface
- [Pandas](https://pandas.pydata.org/) for data processing

## Author

Created by [Josh Copp](https://github.com/copp1723)