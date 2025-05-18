## Chainlit Voice Assistant with Web Search and RAG

This is a multimodal Chainlit application that provides:

1. **Voice Interface**:
   - Press 'P' to talk
   - Automatic speech-to-text conversion
   - Response in both text and voice

2. **Web Search**:
   - Use `/search your query` to search the web
   - Or say "search for..." when using voice input

3. **Document Analysis (RAG)**:
   - Upload PDF and DOCX files
   - Ask questions about the documents
   - Combines search and document knowledge

4. **Multilingual Support**:
   - Select from 50+ languages
   - Both input and output in the selected language

5. **Customization**:
   - Adjust response temperature
   - Toggle streaming mode

### Setup Instructions

1. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   SUTRA_API_KEY=your_sutra_api_key
   LINKUP_API_KEY=your_linkup_api_key
   ```

2. Install required packages:
   ```
   pip install chainlit openai litellm linkup-sdk langchain langchain-openai faiss-cpu python-dotenv numpy
   ```

3. Run the application:
   ```
   chainlit run app.py --port 8477
   ```

### Usage

- **Voice Input**: Press 'P' and speak
- **Text Input**: Type normally in the chat
- **Web Search**: Type `/search your query`
- **Upload Documents**: Use the upload button
- **Language Selection**: Use the dropdown in settings

This app demonstrates integration of multiple AI services (OpenAI for STT/TTS, Sutra for LLM, Linkup for search) in a unified conversational interface.