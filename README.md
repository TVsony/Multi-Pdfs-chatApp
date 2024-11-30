# Multi-PDF-s 📚ChatApp AI Agent 🤖
Meet MultiPDF Chat AI App! 🚀 Chat seamlessly with Multiple PDFs using Langchain, Google Gemini Pro & FAISS Vector DB with Seamless Streamlit Deployment. Get instant, Accurate responses from Awesome Google Gemini OpenSource language Model. 📚💬 Transform your PDF experience now! 🔥✨

## 📝 Description
The Multi-PDF's Chat Agent is a Streamlit-based web application designed to facilitate interactive conversations with a chatbot. The app allows users to upload multiple PDF documents, extract text information from them, and train a chatbot using this extracted content. Users can then engage in real-time conversations with the chatbot.

## 🎯 How It Works:

![alt text](image.png)


### The application follows these steps to provide responses to your questions:

PDF Loading : The app reads multiple PDF documents and extracts their text content.

Text Chunking : The extracted text is divided into smaller chunks that can be processed effectively.

Language Model : The application utilizes a language model to generate vector representations (embeddings) of the text chunks.

Similarity Matching : When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

Response Generation : The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.


## 🎯 Key Features 

Adaptive Chunking: Our Sliding Window Chunking technique dynamically adjusts window size and position for RAG, balancing fine-grained and coarse-grained data access based on data complexity and context.

Multi-Document Conversational QA: Supports simple and multi-hop queries across multiple documents simultaneously, breaking the single-document limitation.

File Compatibility: Supports both PDF and TXT file formats.

LLM Model Compatibility: Supports Google Gemini Pro, OpenAI GPT 3, Anthropic Claude, Llama2 and other open-source LLMs.


## 🌟 Requirements
Streamlit : A Python library for building web applications with interactive elements.
google-generativeai : It is a package that provides generative AI capabilities for chatbots and virtual agents. It can be used in a variety of applications such as content generation, dialogue agents, summarization and classification systems and more.
python-dotenv : A library for loading environment variables from a .env file. This is commonly used to store configuration settings, API keys, and other sensitive information outside of your code.
langchain : A custom library for natural language processing tasks, including conversational retrieval, text splitting, embeddings, vector stores, chat models, and memory.
PyPDF2 : PyPDF2 is a library for reading and manipulating PDF files in Python. It can be used to merge, split, and modify PDFs. In the context of a multipdf chatbot, PyPDF2 could be used to handle the manipulation and generation of PDF files based on user input or responses.
faiss-cpu : FAISS (Facebook AI Similarity Search) is a library developed by Facebook for efficient similarity search, Machine Learning Embeddings,Information Retrieval, content-based filtering and clustering of dense vectors.
langchain_google_genai : It is a package that provides an integration between LangChain and Google’s generative-ai SDK. It contains classes that extend the Embeddings class and provide methods for generating embeddings. The package can be used in a multipdf chatbot application to extract textual data from PDF documents and generate Accurate responses to user queries.

![alt text](image-1.png)


