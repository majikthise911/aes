# Streamlit Cloud Deployment Guide

## Prerequisites
- GitHub account
- Streamlit Cloud account (sign up at share.streamlit.io)
- OpenAI API key

## Deployment Steps

### 1. Push Code to GitHub
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select repository: `your-username/aes`
4. Branch: `main`
5. Main file path: `app.py`

### 3. Configure Secrets

In Streamlit Cloud > App Settings > Secrets, add:

```toml
OPENAI_API_KEY = "sk-your-actual-openai-api-key-here"
```

**Important:** Never commit your API key to the repository!

### 4. Advanced Settings (Optional)

- Python version: 3.11 (recommended)
- Memory: 1GB (default should work)

### 5. Deploy

Click "Deploy" and wait 2-3 minutes for the app to start.

## File Structure

```
aes/
├── app.py                  # Main application entry point
├── config.py              # Configuration and data persistence
├── utils.py               # Utility functions
├── views.py               # UI components and pages
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── config/
│   └── .env              # Local API key (NOT committed)
├── src/
│   ├── data_manager.py   # Data management
│   ├── pdf_processor.py  # PDF extraction
│   ├── gpt_extractor.py  # AI extraction
│   ├── chatbot.py        # AI chatbot
│   └── command_ai.py     # Natural language commands
├── data/
│   ├── proposals.json    # Proposal data storage
│   └── proposals/        # PDF storage
└── Screenshot 2025-10-07 090943.png  # Logo

```

## Environment Variables

The app uses these environment variables:
- `OPENAI_API_KEY` - Your OpenAI API key (required)

## Troubleshooting

### App won't start
- Check that all files are committed and pushed
- Verify requirements.txt is present
- Check Streamlit Cloud logs for errors

### API errors
- Verify OPENAI_API_KEY is set correctly in Secrets
- Make sure API key is active and has credits

### File upload issues
- Streamlit Cloud has a 200MB upload limit (configured in config.toml)
- Large PDFs may take time to process

## Local Development

To run locally:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `config/.env` with:
```
OPENAI_API_KEY=your-api-key-here
```

3. Run the app:
```bash
streamlit run app.py
```

## Updates

To update the deployed app:
```bash
git add .
git commit -m "Update description"
git push origin main
```

Streamlit Cloud will automatically redeploy.

## Security Notes

- ✅ `.env` file is in `.gitignore` - API keys are NOT committed
- ✅ Secrets are managed through Streamlit Cloud interface
- ✅ PDF files are stored locally, not committed to repo
- ✅ Proposal data (proposals.json) is in `.gitignore` for privacy
