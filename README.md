# Stock Research Portfolio Assistant

An AI-powered assistant designed to help with stock research and portfolio management. The application uses a combination of Retrieval-Augmented Generation (RAG) and intelligent agents to provide insights.

## Features
- **Stock Research Agent**: Analyzes stock data and news to provide actionable insights.
- **RAG-based Knowledge Base**: Uses vector search to retrieve relevant information from a localized knowledge base.
- **Modern Web Interface**: A clean and responsive UI for interacting with the assistant.
- **Python Backend**: Robust backend powered by Python to manage agents, tools, and the API server.

## Project Structure
- `agents.py`: Contains the definitions for the intelligent agents.
- `rag.py`: Implementation of the Retrieval-Augmented Generation (RAG) pipeline.
- `tools.py`: Utility functions and tools used by the agents.
- `server.py`: The main entry point for the backend server.
- `knowledge_base/`: Directory containing documents and glossary files used for RAG.
- `frontend/`: Contains the frontend assets (`index.html`, `styles.css`, `app.js`).

## Setup and Installation

### Prerequisites
- Python 3.x
- Virtual environment (recommended)

### Installation
1. Clone the repository and navigate into the directory:
   ```bash
   git clone https://github.com/Sanjana2936/Stock-Research-Portfolio-Assistant.git
   cd Stock-Research-Portfolio-Assistant
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```
3. Install the dependencies (ensure you have a `requirements.txt` file):
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables:
   - Create a `.env` file in the root directory.
   - Add your necessary API keys (e.g., LLM provider keys).

### Running the Application
1. Start the backend server:
   ```bash
   python server.py
   ```
2. Open `frontend/index.html` in your browser or run a local development server for the frontend.

## License
[MIT License](LICENSE)
