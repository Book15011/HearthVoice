import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key or api_key == 'your-api-key-here':
    print("Error: DEEPSEEK_API_KEY not found or not set.")
    print("Please make sure you have a .env file with your key:")
    print("DEEPSEEK_API_KEY='your-real-api-key'")
else:
    print("API key found. Testing connection to DeepSeek...")
    try:
        # Initialize the ChatDeepseek model
        llm = ChatDeepSeek(model="deepseek-chat")

        # Send a test message
        response = llm.invoke("好開心識你，你想唔想同我食晚飯?")

        # Print the response
        print("\nDeepSeek responded:")
        print(response.content)
        print("\nTest successful! Your DeepSeek connection is working.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("\nTest failed. Please check your API key and network connection.")
