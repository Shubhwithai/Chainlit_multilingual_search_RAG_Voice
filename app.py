import os
import io
import wave
import numpy as np
import audioop
from dotenv import load_dotenv
import chainlit as cl
import json
import asyncio
from chainlit.input_widget import Select, Switch, Slider
from datetime import datetime

# LLM and API imports
from openai import AsyncOpenAI as AsyncOpenAIClient
from openai import AsyncOpenAI as AsyncSutraClient
import litellm
from linkup import LinkupClient

# RAG imports
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

# API keys and configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUTRA_API_KEY = os.getenv("SUTRA_API_KEY")
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY")

# API base for Sutra
API_BASE = "https://api.two.ai/v2"

# Client initializations
openai_client = AsyncOpenAIClient(api_key=OPENAI_API_KEY)
sutra_client = AsyncSutraClient(
    base_url=API_BASE,
    api_key=SUTRA_API_KEY
)
linkup_client = LinkupClient(api_key=LINKUP_API_KEY)

# Audio and silence detection settings
SILENCE_THRESHOLD = 1500
SILENCE_TIMEOUT = 1800.0

# Available languages
LANGUAGES = [
    "English", "Hindi", "Gujarati", "Bengali", "Tamil", "Telugu", "Kannada", "Malayalam",
    "Punjabi", "Marathi", "Urdu", "Assamese", "Odia", "Sanskrit", "Korean", "Japanese",
    "Arabic", "French", "German", "Spanish", "Portuguese", "Russian", "Chinese",
    "Vietnamese", "Thai", "Indonesian", "Turkish", "Polish", "Ukrainian", "Dutch",
    "Italian", "Greek", "Hebrew", "Persian", "Swedish", "Norwegian", "Danish",
    "Finnish", "Czech", "Hungarian", "Romanian", "Bulgarian", "Croatian", "Serbian",
    "Slovak", "Slovenian", "Estonian", "Latvian", "Lithuanian", "Malay", "Tagalog", "Swahili"
]

# Commands for UI
COMMANDS = [
    {
        "id": "Search",
        "icon": "globe",
        "description": "Find on the web",
        "button": True,
        "persistent": True
    },
]

# Process uploaded documents for RAG
def process_documents(files, chunk_size=1000, chunk_overlap=100):
    """Process uploaded documents for RAG"""
    documents = []
    pdf_elements = []
    
    for file in files:
        if file.name.endswith(".pdf"):
            pdf_elements.append(
                cl.Pdf(name=file.name, display="side", path=file.path)
            )
            loader = PyPDFLoader(file.path)
            documents.extend(loader.load())
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file.path)
            documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    document_chunks = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(document_chunks, embeddings)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            api_key=SUTRA_API_KEY,
            base_url=API_BASE,
            model="sutra-v2",
            streaming=False
        ),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return conversation_chain, pdf_elements

# Search web using Linkup
async def search_web(query: str, depth: str) -> str:
    """Search the web using Linkup SDK"""
    try:
        search_results = linkup_client.search(
            query=query,
            depth=depth,
            output_type="searchResults",
        )

        formatted_text = "Search results:\n\n"

        for i, result in enumerate(search_results.results, 1):
            formatted_text += f"{i}. **{result.name}**\n"
            formatted_text += f"   URL: {result.url}\n"
            formatted_text += f"   {result.content}\n\n"

        return formatted_text
    except Exception as e:
        return f"Search failed: {str(e)}"

# Chat initialization
@cl.on_chat_start
async def start():
    # Initialize chat settings
    await cl.ChatSettings([
        Select(id="language", label="🌐 Language", values=LANGUAGES, initial_index=0),
        Switch(id="streaming", label="💬 Stream Response", initial=True),
        Slider(id="temperature", label="🔥 Temperature", initial=0.7, min=0, max=1, step=0.1),
    ]).send()

    # Set up commands
    await cl.context.emitter.set_commands(COMMANDS)
    
    # Initialize session data
    cl.user_session.set("documents_processed", False)
    cl.user_session.set("conversation_chain", None)
    cl.user_session.set("pdf_elements", [])
    cl.user_session.set("chat_messages", [])
    cl.user_session.set("message_history", [])
    
    # Audio session variables
    cl.user_session.set("silent_duration_ms", 0)
    cl.user_session.set("is_speaking", False)
    cl.user_session.set("audio_chunks", [])
    
    # Welcome message
    welcome_text = "🎤 Welcome! Press `P` to talk, upload documents for RAG, or type a message."
    msg = cl.Message(content="")
    await msg.send()
    for token in welcome_text:
        await msg.stream_token(token)
        await asyncio.sleep(0.005)

# Speech-to-Text conversion
@cl.step(type="tool")
async def speech_to_text(audio_file):
    response = await openai_client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_file,
        language="en"
    )
    return response.text

# Text-to-Speech conversion
@cl.step(type="tool")
async def text_to_speech(text: str):
    response = await openai_client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="nova",
        input=text,
    )
    buffer = io.BytesIO()
    buffer.name = "response.wav"
    buffer.write(response.content)
    buffer.seek(0)
    return buffer.name, buffer.read()

# Generate response using Sutra
async def generate_response(prompt, is_rag=False, search_result=None):
    settings = cl.user_session.get("chat_settings")
    language = settings.get("language", "English")
    temperature = settings.get("temperature", 0.7)
    
    # Get current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare messages for the model
    messages = []
    
    # Add system message based on context
    if is_rag:
        # If documents are processed, use RAG
        conversation_chain = cl.user_session.get("conversation_chain")
            
        # Get RAG context
        rag_response = conversation_chain.invoke(prompt)
        context = rag_response["answer"]
        
        # If search result is provided, combine RAG with web search
        if search_result:
            context = f"{context}\n\nAdditional web search results:\n{search_result}"
            
        system_prompt = f"""
        You are a helpful assistant that answers questions about documents and web content. 
        Use ONLY the following context to answer the question. DO NOT use any external knowledge.
        
        Current Date and Time: {current_datetime}
        
        CONTEXT:
        {context}
        
        IMPORTANT INSTRUCTIONS:
        1. You MUST respond in {language} language only.
        2. You MUST ONLY use information from the provided context above.
        3. If the answer is not in the context, say "I cannot find this information in the provided documents."
        4. DO NOT make assumptions or add information not present in the context.
        5. Translate all content, including document content, search results, and sources, into {language}.
        6. At the end, list all the sources used with their names and URLs.
        7. Keep URLs unchanged while translating other content.
        8. If the document content is in a different language, translate it to {language} in your response.
        """
    elif search_result:
        # Direct chat with search results
        system_prompt = f"""
        You are a helpful assistant. Use the following search results to answer the question.
        
        Current Date and Time: {current_datetime}
        
        SEARCH RESULTS:
        {search_result}
        
        IMPORTANT: You MUST respond in {language} language only. Translate all content, including search results and sources, into {language}.
        Please provide a comprehensive answer and at the end, list all the sources used with their names and URLs.
        Make sure to translate the source names and content into {language} while keeping the URLs unchanged.
        """
    else:
        # Direct chat without RAG or search
        system_prompt = f"""
        You are a helpful assistant using the Sutra-v2 model.
        Current Date and Time: {current_datetime}
        
        IMPORTANT: You MUST respond in {language} language only.
        Please provide a comprehensive answer. Focus on being helpful, concise, and accurate.
        """
    
    # Add system message
    messages.append({"role": "system", "content": system_prompt})
    
    # Add user message
    messages.append({"role": "user", "content": prompt})
    
    # Generate response
    response = await sutra_client.chat.completions.create(
        model="sutra-v2",
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
        stream=False
    )

    return response.choices[0].message.content

# Audio start handler
@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("silent_duration_ms", 0)
    cl.user_session.set("is_speaking", False)
    cl.user_session.set("audio_chunks", [])
    return True

# Audio chunk processor
@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    audio_chunks = cl.user_session.get("audio_chunks")
    if audio_chunks is not None:
        audio_chunks.append(np.frombuffer(chunk.data, dtype=np.int16))

    if chunk.isStart:
        cl.user_session.set("last_elapsed_time", chunk.elapsedTime)
        cl.user_session.set("is_speaking", True)
        return

    time_diff = chunk.elapsedTime - cl.user_session.get("last_elapsed_time")
    cl.user_session.set("last_elapsed_time", chunk.elapsedTime)

    energy = audioop.rms(chunk.data, 2)
    if energy < SILENCE_THRESHOLD:
        silent_duration = cl.user_session.get("silent_duration_ms") + time_diff
        cl.user_session.set("silent_duration_ms", silent_duration)
        if silent_duration >= SILENCE_TIMEOUT and cl.user_session.get("is_speaking"):
            cl.user_session.set("is_speaking", False)
            await process_audio()
    else:
        cl.user_session.set("silent_duration_ms", 0)
        cl.user_session.set("is_speaking", True)

# Process audio for transcription and response
async def process_audio():
    audio_chunks = cl.user_session.get("audio_chunks", [])
    if not audio_chunks:
        return

    combined = np.concatenate(audio_chunks)
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(combined.tobytes())

    wav_buffer.seek(0)
    cl.user_session.set("audio_chunks", [])

    audio_file = ("audio.wav", wav_buffer, "audio/wav")
    transcription = await speech_to_text(audio_file)

    user_message = cl.Message(
        author="You",
        type="user_message",
        content=transcription,
        elements=[cl.Audio(content=wav_buffer.getvalue(), mime="audio/wav")]
    )
    await user_message.send()

    # Check for search intent in transcription
    search_needed = any(keyword in transcription.lower() for keyword in ["search", "find", "look up", "google", "web", "internet"])
    
    # Process based on context
    is_rag = cl.user_session.get("documents_processed", False)
    search_result = None
    
    # If search is explicitly requested or detected
    if search_needed:
        search_result = await search_web(transcription, "standard")
    
    # Generate response based on context
    reply = await generate_response(transcription, is_rag, search_result)
    
    # Convert to speech
    _, output_audio = await text_to_speech(reply)
    
    # Create PDF elements if available
    elements = [cl.Audio(content=output_audio, mime="audio/wav", auto_play=True)]
    pdf_elements = cl.user_session.get("pdf_elements", [])
    if pdf_elements:
        elements.extend(pdf_elements)
    
    # Send response
    await cl.Message(
        content=reply,
        elements=elements
    ).send()
    
    # Update chat history
    chat_messages = cl.user_session.get("chat_messages", [])
    chat_messages.append({"role": "user", "content": transcription})
    chat_messages.append({"role": "assistant", "content": reply})
    cl.user_session.set("chat_messages", chat_messages)

# Message handler (for text input)
@cl.on_message
async def on_message(msg: cl.Message):
    """Handle incoming user messages"""
    settings = cl.user_session.get("chat_settings")
    language = settings.get("language", "English")
    temperature = settings.get("temperature", 0.7)
    streaming = settings.get("streaming", True)
    
    # Handle file uploads if present
    if hasattr(msg, 'elements') and msg.elements:
        try:
            files = [element for element in msg.elements if element.type == "file"]
            if files:
                # Process files immediately
                conversation_chain, pdf_elements = process_documents(files)
                cl.user_session.set("conversation_chain", conversation_chain)
                cl.user_session.set("documents_processed", True)
                cl.user_session.set("pdf_elements", pdf_elements)
                
                if pdf_elements:
                    await cl.Message(content="✅ Documents processed successfully.", elements=pdf_elements).send()
                else:
                    await cl.Message(content="✅ Documents processed successfully.").send()
                
                # If no message content, return after processing files
                if not msg.content:
                    return
        except Exception as e:
            if "API key" in str(e):
                await cl.Message(content="Please check your API keys in the environment variables.").send()
            return
    
    # If no message content and no files, return
    if not msg.content:
        return
    
    # Get chat history
    chat_messages = cl.user_session.get("chat_messages", [])
    chat_messages.append({"role": "user", "content": msg.content})

    # Generate response
    response = cl.Message(content="")
    await response.send()
    
    try:
        # Check for search command
        search_result = None
        if msg.command == "Search":
            search_result = await search_web(msg.content, "standard")
        
        # Check if RAG is active
        is_rag = cl.user_session.get("documents_processed", False)
        
        # Generate response based on context
        reply = await generate_response(msg.content, is_rag, search_result)
        
        # Stream response tokens if streaming is enabled
        if streaming:
            for char in reply:
                await response.stream_token(char)
                await asyncio.sleep(0.0025)
        else:
            await response.stream_token(reply)
        
        # Generate speech for the response
        _, output_audio = await text_to_speech(reply)
        
        # Add PDF elements to the response if available
        elements = [cl.Audio(content=output_audio, mime="audio/wav", auto_play=True)]
        pdf_elements = cl.user_session.get("pdf_elements", [])
        if pdf_elements:
            elements.extend(pdf_elements)
            
        # Create a new message with the final content and elements
        final_response = cl.Message(content=response.content, elements=elements)
        await final_response.send()
        # Remove the streaming message
        await response.remove()
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        await response.stream_token(error_msg)
        if "API key" in str(e):
            await cl.Message(content="Please check your API keys in the environment variables.").send()
        return

    # Update chat history
    chat_messages.append({"role": "assistant", "content": reply})
    cl.user_session.set("chat_messages", chat_messages)

# Resume chat session
@cl.on_chat_resume
async def on_chat_resume(thread):
    """Resume chat session"""
    # Restore user session data
    cl.user_session.set("documents_processed", thread.get("documents_processed", False))
    cl.user_session.set("conversation_chain", thread.get("conversation_chain"))
    cl.user_session.set("pdf_elements", thread.get("pdf_elements", []))
    cl.user_session.set("chat_messages", thread.get("chat_messages", []))
    
    # Show welcome back message
    await cl.Message(
        content="Welcome back! Press `P` to talk, type a message, or upload documents."
    ).send()
