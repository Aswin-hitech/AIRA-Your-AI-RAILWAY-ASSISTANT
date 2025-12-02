# AIRA - AI Railway Assistant

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Tech Stack](#tech-stack)
6. [Installation & Setup](#installation--setup)
7. [Configuration](#configuration)
8. [Usage Guide](#usage-guide)
9. [Module Documentation](#module-documentation)
10. [CSV Database Setup](#csv-database-setup)
11. [SMS Integration](#sms-integration)
12. [Future Enhancements](#future-enhancements)
13. [Troubleshooting](#troubleshooting)
14. [Contributors](#contributors)

---

## Project Overview

**AIRA** (AI Railway Assistant) is an intelligent railway complaint management system that leverages artificial intelligence and machine learning to streamline passenger complaint handling and emergency response in trains. It uses natural language processing, voice recognition, and ML classification to categorize and prioritize complaints, routing them to the appropriate authorities (police, medical teams, technical staff) based on urgency and complaint type.

The system processes complaints through a conversational chatbot interface, either text-based or voice-enabled, and intelligently routes high-urgency complaints via SMS notifications to relevant railway personnel for faster emergency response.

---

## Features

### Core Features:

- **PNR Integration**: Fetch train, coach, and seat details directly from PNR using web scraping
- **Voice-to-Text Conversion**: Accept complaints via voice input and convert to text using speech recognition
- **Intelligent Complaint Classification**: Classify complaints into Management, Technical, or Other Issue categories using DistilBERT model
- **Urgency Detection**: Detect complaint urgency level (High/Medium) using BERT-based NLP model with critical keyword matching
- **Medical Emergency Detection**: Automatically identify medical issues from complaints and route to medical teams
- **Smart Routing**: Direct complaints to appropriate authorities (Police, Medical Teams, Technical Staff)
- **SMS Notifications**: Send real-time alerts to police officers and medical teams for high-urgency complaints
- **Complaint History**: Maintain comprehensive CSV logs of all complaints with metadata
- **Web Interface**: User-friendly Flask-based chat interface with conversation history and navigation

### Data Processing Capabilities:

- Text cleaning and preprocessing using regex patterns
- CSV data management with pandas
- Model training with PyTorch and Hugging Face transformers
- Confusion matrix visualization and performance metrics

---

## Architecture

### Component Hierarchy:

```
Frontend (Flask Web Interface)
    ├── Chat Interface (chat.html)
    ├── Home Page (index.html)
    └── Voice Recorder UI

    ↓

Backend Processing Layer
    ├── app.py (Main Flask Application)
    ├── Session Management
    └── State Machine (Multi-turn conversation flow)

    ↓

NLP & ML Pipeline
    ├── Urgency_finder.py (BERT-based urgency classifier)
    ├── Management_technical_split.py (DistilBERT issue classifier)
    ├── voice_to_text.py (Speech Recognition)
    ├── Words_correction.py (Text normalization)
    └── medical_finder.py (Medical keyword matching)

    ↓

Data Extraction & Integration
    ├── getpnr.py (Web scraping for PNR details)
    ├── police_complaint_details.py (Database lookup)
    ├── Complaint_input_file.py (Data preparation)
    └── sms_sender.py (Twilio SMS gateway)

    ↓

Storage Layer
    ├── Police_id.csv (Officer & Medical contact database)
    ├── issue_classify.csv (Training data for issue classification)
    ├── railwaytest.csv (Training data for urgency detection)
    └── Save_to_CSV.csv (Complaint logs)
```

### Data Flow:

1. **Input**: User provides PNR or manual train/coach/seat details
2. **Extraction**: System fetches booking details from RailRestro
3. **Complaint**: User describes issue (text or voice)
4. **Analysis**: NLP models classify urgency and category
5. **Routing**: System identifies relevant personnel from database
6. **Notification**: SMS alerts sent to authorities
7. **Logging**: Complaint saved to CSV with metadata

---

## Project Structure

```
AIRA/
├── Backend/
│   ├── __init__.py
│   ├── __pycache__/
│   ├── main.py                         # Alternative CLI entry point
│   ├── Complaint_input_file.py         # Data preparation module
│   ├── getpnr.py                       # PNR web scraping
│   ├── Urgency_finder.py               # Urgency classification (BERT)
│   ├── Management_technical_split.py   # Issue classification (DistilBERT)
│   ├── medical_finder.py               # Medical issue detection
│   ├── police_complaint_details.py     # Police contact lookup
│   ├── sms_sender.py                   # Twilio SMS integration
│   ├── voice_to_text.py                # Speech-to-text conversion
│   ├── Words_correction.py             # Text normalization utilities
│   ├── twi.env                         # Twilio credentials (confidential)
│
├── static/
│   ├── Assets/
│   │   ├── train1.jpg to train6.avif   # Train images for UI
│   ├── css/                            # Stylesheets
│   ├── js/
│   │   └── scripts.js                  # Frontend JavaScript
│   └── audio/                          # Voice recordings storage
│
├── templates/
│   ├── index.html                      # Homepage
│   ├── chat.html                       # Chat interface
│
├── Storage files/
│   ├── Police_id.csv                   # Police/Medical contact database
│   ├── issue_classify.csv              # Issue classification training data
│   ├── railwaytest.csv                 # Urgency detection training data
│   ├── Save_to_CSV.csv                 # Complaint logs
│   └── [Additional CSV files]
│
├── checkpoints/                        # Model training checkpoints
├── checkpoints_issue/                  # Issue model checkpoints
├── issue_classify_model/               # Saved DistilBERT model
├── urgency_model/                      # Saved BERT model
├── logs/                               # Training logs
├── logs_issue/                         # Issue training logs
├── metrics/                            # Performance metrics
│
|── app.py                          # Main Flask application
└── graph.py                            # Data visualization utilities
```

---
## GITIGNORED FILES

Backend/twi.env
Storage files/Police_id.csv
Storage files/issue_classify.csv
Storage files/railwaytest.csv
issue_classify_model/
urgency_model/
checkpoints/
checkpoints_issue/
static/audio/

## You must add your own copies when running the project locally.


## Tech Stack

### Backend Framework:
- **Flask 2.x** - Web application framework
- **Python 3.8+** - Programming language

### Machine Learning & NLP:
- **PyTorch** - Deep learning framework
- **Hugging Face Transformers** - Pre-trained BERT & DistilBERT models
- **scikit-learn** - Metrics and preprocessing utilities
- **NLTK** - Natural language toolkit

### Data Processing:
- **Pandas** - Data manipulation and CSV handling
- **NumPy** - Numerical computations

### Voice Processing:
- **SpeechRecognition** - Audio-to-text conversion (Google Speech API)
- **PyDub** - Audio format conversion
- **FFmpeg** - Audio codec support

### Web Scraping & HTTP:
- **BeautifulSoup4** - HTML parsing
- **Requests** - HTTP client
- **Selenium** (optional) - Browser automation

### SMS Integration:
- **Twilio** - SMS gateway service

### Frontend:
- **HTML5/CSS3** - UI markup and styling
- **JavaScript (Vanilla)** - Frontend interactivity
- **WebRTC** - Voice recording API

### Database:
- **CSV files** - Data storage (Pandas-managed)

### Visualization:
- **Matplotlib** - Chart plotting
- **Seaborn** - Statistical visualizations

---

## Installation & Setup

### Prerequisites:

- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)
- Twilio account (for SMS notifications)
- FFmpeg installed on system


### Step 1: Clone Repository

```bash
git clone https://github.com/your-repo/AIRA.git
cd AIRA
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install FFmpeg

```bash
# Windows (using choco)
choco install ffmpeg

# macOS (using brew)
brew install ffmpeg

# Linux (Ubuntu/Debian)
sudo apt-get install ffmpeg
```

### Step 5: Create Storage Directories

```bash
mkdir -p "Storage files"
mkdir -p static/audio
mkdir -p logs
```

### Step 6: Download Pre-trained Models

The models will auto-download on first run from Hugging Face Hub:
- `bert-base-uncased` (for urgency detection)
- `distilbert-base-uncased` (for issue classification)

---

## Configuration

### Environment Variables (twi.env):

Create a `Backend/twi.env` file with your Twilio credentials:

```
TWILIO_SID=your_account_sid_here
TWILIO_AUTH=your_auth_token_here
TWILIO_PHONE=+1234567890
## ⚠️ Twilio SMS costs money after trial credits. Disable the SMS feature if using in a demo.
```

**How to get Twilio credentials:**
1. Sign up at [twilio.com](https://www.twilio.com)
2. Get Account SID from Dashboard
3. Get Auth Token from Dashboard
4. Rent a phone number from Twilio
5. Add credentials to `twi.env`

### Flask Configuration (app.py):

```python
app.secret_key = "secret_key_for_session"  # Change to secure key in production
app.run(host="0.0.0.0", port=5000, debug=True)
```

### Model Training Configuration:

**Urgency Model** (Urgency_finder.py):
- Model: BERT (bert-base-uncased)
- Max Length: 128 tokens
- Batch Size: 16
- Epochs: 10
- Learning Rate: 2e-5

**Issue Classifier** (Management_technical_split.py):
- Model: DistilBERT (distilbert-base-uncased)
- Max Length: 64 tokens
- Batch Size: 16
- Epochs: 8
- Classes: Management Issue, Technical Issue, Other Issue

---

## Usage Guide

### Running the Application:

```bash
# Navigate to Backend directory
cd Backend

# Run Flask server
python app.py

# Server starts on http://localhost:5000
```

### User Workflow:

1. **Visit Application**: Open `http://localhost:5000` in browser
2. **Start Chat**: Click "Chat with AIRA"
3. **Provide Train Details**:
   - Option A: Enter 10-digit PNR (system auto-fetches train info)
   - Option B: Manually enter Train Number, Coach, Seat
4. **Describe Complaint**: Text or voice input
5. **Review Analysis**: AIRA classifies urgency and category
6. **Automatic Notification**: High-urgency complaints trigger SMS alerts
7. **Track History**: All complaints logged to CSV

### Example Complaint Flow:

```
Bot: "Welcome to AIRA! Do you have a PNR number?"
User: "Yes"
Bot: "Please enter your 10-digit PNR:"
User: "1234567890"
Bot: "Fetched details. Train: 12345, Coach: A, Seat: 12. Correct?"
User: "Yes"
Bot: "Please describe your complaint:"
User: "Fire in the toilet near my seat!"
Bot: "Category: Technical Issue | Urgency: High"
Bot: "Alert sent to Officer Raj Kumar, Mobile: +919876543210"
Bot: "Complaint registered! Need help with another issue?"
```

---

## Module Documentation

### app.py

Main Flask application handling HTTP routes and session management.

**Key Routes:**
- `GET /` - Homepage
- `GET /chat_page` - Chat interface
- `GET /chat_default` - Initial greeting
- `POST /chat` - Main chat logic
- `POST /voice_to_text` - Voice processing

**Key Features:**
- Multi-turn conversation state machine
- Session-based user context persistence
- Complaint data persistence to CSV
- Integration with all backend modules

### Urgency_finder.py

BERT-based urgency detection with critical keyword matching.

**Functions:**
- `check_urgency(text)` - Returns "High" or "Medium" urgency
- `train_and_evaluate()` - Model training with evaluation metrics
- `clean_text(text)` - Text preprocessing

**Critical Keywords:** Fire, explosion, medical emergency, assault, derailment, etc.

### Management_technical_split.py

DistilBERT-based complaint classification.

**Functions:**
- `classify_issue(complaint_text)` - Returns issue category
- `train_and_evaluate()` - Model training with confusion matrix
- `compute_metrics(eval_pred)` - Performance evaluation

**Classes:** Management Issue (0), Technical Issue (1), Other Issue (2)

### medical_finder.py

Medical emergency detection and contact routing.

**Functions:**
- `is_medical_issue(complaint_text)` - Detects medical keywords
- `medical_complaint(train, coach, seat, complaint_text, urgency, police_df)` - Gets medical contact
- `format_phone(number)` - Formats phone numbers to +91 format

**Medical Keywords:** Heart attack, injury, bleeding, ambulance, emergency, paramedics, etc.

### voice_to_text.py

Speech recognition with audio format conversion.

**Functions:**
- `convert_voice_to_text(base64_audio)` - Converts base64 audio to text
- `get_next_audio_filename()` - Generates sequential filename

**Supported Formats:** WebM (input), WAV (processing)

### police_complaint_details.py

Police officer and medical team contact lookup from CSV database.

**Functions:**
- `police_complaint(train, coach, seat)` - Returns officer name and phone
- `format_phone(number)` - Ensures +91 format

**During process add your mobile numbers in CSV database for working as well**

### getpnr.py

Web scraping PNR details from RailRestro website.

**Functions:**
- `extract_coach_train_seat_from_railrestro(pnr)` - Returns (train, coach, seat)

**Uses:** BeautifulSoup4 for HTML parsing, regex for data extraction

### sms_sender.py

Twilio SMS integration for sending alerts.

**Functions:**
- `send_sms(to_number, message)` - Send SMS to single recipient
- `send_sms_to_multiple(numbers, message)` - Send to multiple recipients

**Requires:** Twilio credentials in twi.env

---

## CSV Database Setup

### Police_id.csv Structure:

```csv
Train Number,Compartment No,Officer Name,Officer Mobile No.,Medical Team
12345,A,Raj Kumar,9876543210,9123456789
12345,B,Priya Singh,9876543211,9123456790
12346,A,Ajay Verma,9876543212,9123456791
```

**Columns:**
- `Train Number`: 5-digit train number
- `Compartment No`: Coach letter (A-F)
- `Officer Name`: Police officer assigned to this compartment
- `Officer Mobile No.`: 10-digit Indian phone number
- `Medical Team`: Contact for medical emergency

**⚠️ IMPORTANT:** Add your mobile numbers to this CSV database for the system to work!

### issue_classify.csv Structure:

```csv
complaints,label
"Seat is broken",0
"Lights not working",1
"Crowded compartment",2
```

**Columns:**
- `complaints`: Complaint text sample
- `label`: Category (0=Management, 1=Technical, 2=Other)

### railwaytest.csv Structure:

```csv
Complaint,Urgency
"Fire in toilet",1
"Missing luggage",0
"Medical emergency",1
```

**Columns:**
- `Complaint`: Sample complaint text
- `Urgency`: Binary (1=High, 0=Medium)

### Save_to_CSV.csv Structure:

```csv
Timestamp,PNR,Train,Coach,Seat,Complaint,Urgency,Issue_Type,Police_Info,Medical_Info
2024-12-02 10:30:45,1234567890,12345,A,12,"Fire in toilet",High,Technical Issue,"Officer: Raj Kumar, Phone: +919876543210",""
```

---

## SMS Integration

### Setting Up Twilio:

1. **Create Twilio Account**: Visit [twilio.com/console/sign-up](https://www.twilio.com/console/sign-up)
2. **Verify Phone Number**: Enter and verify your mobile number
3. **Get Credentials**:
   - Account SID: From Dashboard
   - Auth Token: From Dashboard
4. **Buy Phone Number**: Purchase a number from Twilio for sending SMS
5. **Configure twi.env**:
   ```
   TWILIO_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   TWILIO_AUTH=your_auth_token_here
   TWILIO_PHONE=+1234567890
   ```

### Testing SMS:

```python
from sms_sender import send_sms

send_sms("+919876543210", "Test message from AIRA!")
```

### SMS Content Example:

```
🚨 URGENT ALERT 🚨
Train: 12345
Coach: A
Seat: 12
Complaint: Fire in toilet
Category: Technical Issue
Urgency: High
```

---

## Future Enhancements

### Planned Features:

- **Real-time Tracking**: Live GPS location of trains
- **Multilingual Support**: Support for Hindi, Tamil, Telugu, Kannada
- **Image Recognition**: Detect issues from photos
- **Incident Analytics**: Dashboard for complaint trends and patterns
- **Mobile App**: Native iOS/Android applications
- **Integration with Indian Railways API**: Direct RailNet integration
- **Predictive Analytics**: Forecast complaint patterns by route/time
- **Advanced NER**: Identify sensitive information in complaints
- **Escalation Workflow**: Multi-level approval for critical issues
- **Feedback System**: Rate and review complaint resolution

---

## Troubleshooting

### Common Issues:

**1. "ModuleNotFoundError: No module named 'transformers'"**
```bash
pip install transformers
```

**2. "FFmpeg not found"**
- Windows: `choco install ffmpeg`
- macOS: `brew install ffmpeg`
- Linux: `sudo apt-get install ffmpeg`

**3. "CUDA Error" (GPU memory)**
- Solution: System will auto-fallback to CPU
- Check: `torch.cuda.is_available()`

**4. "Twilio credentials missing"**
- Verify `Backend/twi.env` exists
- Check credentials are correct
- Reload environment: `load_dotenv()`

**5. "PNR extraction failed"**
- PNR might be invalid
- RailRestro website structure changed
- Check network connection
- Verify BeautifulSoup selectors in `getpnr.py`

**6. "Speech Recognition not working"**
- Ensure microphone permissions granted
- Check internet (Google Speech API needs connection)
- Try WAV format audio files
- Install: `pip install SpeechRecognition`

**7. "Model loading slow first time"**
- Models download from Hugging Face (~500MB)
- Patience required for first run
- Models cached after first download
- Consider pre-downloading: `python -m transformers-cli download`

---

## Contributors

- **Project Guide**: Ms. Anu Priya A
- **Development Team**: Aswin N , Melvin Jessan , ManojKumar C

---

## License

This project is developed for academic purposes. Modify and distribute as per institutional guidelines.

---

## Contact & Support

For issues, suggestions, or contributions:
- Create GitHub Issues
- Email: kit28.24bam009@gmail.com
         kit28.24bam041@gmail.com
         kit28.24bam039@gmail.com

---

**Last Updated**: December 2025
**Version**: 1.0.0  
**Status**: Active Development
