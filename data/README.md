# Data Directory

This directory stores all proposal data and uploaded files.

## Structure

```
data/
├── proposals/          # Uploaded PDF files
│   └── YYYYMMDD_HHMMSS_filename.pdf
├── proposals.json      # Extracted proposal data
└── README.md          # This file
```

## Files

- **proposals.json**: Contains all extracted data from proposals (costs, equipment, scope, etc.)
- **proposals/**: Contains original PDF files with timestamp prefixes to avoid naming conflicts

## Notes

- PDF files are automatically saved when uploaded through the app
- Files are named with timestamps: `20250130_143022_proposal.pdf`
- The `proposals.json` file is automatically updated when proposals are added or removed
- When you click "Clear All Data", both the JSON file and all PDFs are deleted
- This folder is excluded from git via .gitignore for privacy

## Re-processing

If scope data is missing from existing proposals, use the "Extract Scope Data" button in the Scope Comparison tab. The app will:
1. Read the stored PDF from `proposals/`
2. Re-extract scope information
3. Update `proposals.json`