# EPC Proposal Dashboard

A Streamlit application for analyzing and visualizing EPC (Engineering, Procurement, Construction) proposals for renewable energy projects.

## Features

- 📁 **File Upload**: Support for PDF proposals and Excel schedule of values
- 🤖 **AI-Powered Extraction**: Uses GPT to extract structured data from inconsistent proposal formats
- 🗺️ **Interactive Maps**: Visualize project locations with detailed popups
- 💰 **Cost Analysis**: Comprehensive cost breakdowns and comparisons
- 📊 **Technology Insights**: Technology type analysis (PV, BESS, Wind, etc.)
- 🔍 **Filtering & Search**: Filter proposals by technology, capacity, location
- 📋 **Comparison View**: Side-by-side comparison of multiple proposals
- 📤 **Export Capabilities**: Export data to CSV and Excel formats

## Installation

1. Clone or download this project
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Copy `config/.env.example` to `config/.env`
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser to the provided URL (typically http://localhost:8501)

3. Upload your EPC proposal PDFs and Excel files using the sidebar

4. Explore the different tabs:
   - **Overview**: High-level metrics and technology breakdown
   - **Map View**: Geographic visualization of projects
   - **Cost Analysis**: Detailed cost comparisons and charts
   - **Comparison**: Side-by-side proposal comparison table

## Project Structure

```
aes/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── config/
│   └── .env.example      # Environment variables template
├── src/
│   ├── pdf_processor.py   # PDF text extraction
│   ├── gpt_extractor.py   # AI-powered data extraction
│   └── data_manager.py    # Data storage and management
├── utils/
│   └── export_utils.py    # Export functionality
└── data/                 # Data storage (created automatically)
```

## Data Extraction

The application uses GPT-4 to extract the following information from EPC proposals:

### Project Information
- Project name
- Location (address, city, county, state, coordinates)
- Jurisdiction

### Technology Details
- Technology type (PV, BESS, Wind, combinations)
- Technology description

### Capacity Information
- AC capacity (MW)
- DC capacity (MW)
- Storage capacity (MWh)

### Equipment Specifications
- Solar modules (manufacturer, model, wattage)
- Inverters (manufacturer, model)
- Racking systems (manufacturer, type)
- Batteries (manufacturer, model, capacity)

### Cost Analysis
- Total project cost
- Cost breakdowns (equipment, labor, materials, development)
- Cost per watt (DC and AC)

## Features in Detail

### AI Data Extraction
- Handles inconsistent proposal formats
- Extracts structured data from unstructured PDFs
- Intelligent location parsing and coordinate extraction

### Interactive Mapping
- Projects displayed on interactive map
- Detailed popups with project information
- Clustered view for multiple projects in same area

### Cost Visualization
- Bar charts for total project costs
- Cost per watt comparisons
- Detailed breakdown tables
- Export capabilities for cost data

### Comparison Tools
- Side-by-side proposal comparison
- Filtering by technology, capacity, location
- Export comparison tables to CSV/Excel

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for map tiles and AI processing

## Troubleshooting

1. **API Key Issues**: Ensure your OpenAI API key is correctly set in the `.env` file
2. **PDF Processing Errors**: Some PDFs may have text extraction issues - try converting to a different PDF format
3. **Memory Issues**: Large PDFs may consume significant memory during processing

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is provided as-is for educational and commercial use.