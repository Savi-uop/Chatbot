AI Food Systems Chatbot
This project is an intelligent RAG (Retrieval-Augmented Generation) chatbot designed to answer questions based on the "Global Food Systems & 2026 AI Compliance Framework."

Overview
The chatbot leverages LangChain and Google's Gemini models to provide insights into AI-driven food innovation, sustainability, and mandatory compliance requirements as outlined in the 2026 AI Ethics & Governance Framework.

Key Features
Document Retrieval: Seamlessly queries the Global Food Systems PDF to provide accurate, context-aware answers.

Compliance Ready: Built with an understanding of Policy #104-B, including data scrubbing and source attribution requirements.

Interactive UI: A professional, chat-based interface built with Streamlit.

Prerequisites
Ensure you have the following installed:

Python 3.10+

Conda (for environment management)

Setup Instructions

1.Clone the repository:
git clone <your-repository-url>
cd Chatbot

2.Set up the virtual environment:
conda create -n chatbot python=3.10
conda activate chatbot

3.Install dependencies:
pip install -r requirements.txt

4.Configure Environment Variables:
Create a .env file in the project root and add your API key:
GOOGLE_API_KEY=your_actual_api_key_here

5.Run the Application:
streamlit run app.py
