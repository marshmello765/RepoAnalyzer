import os
import tempfile
from threading import main_thread
from dotenv import load_dotenv
import streamlit as st
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from utils import format_user_question
from file_processing import clone_github_repo, load_and_index_files
from questions import ask_question, QuestionContext
import time

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_name = "gpt-3.5-turbo"

def main():
    st.title("GITGPT😎")
    st.subheader("A GitHub assistant powered by GPT-3")
    progress_text = "Operation in progress. Please wait."
    github_url = st.text_input("Enter the GitHub URL of the repository: ")
    if st.button("Run analysis"):
        repo_name = github_url.split("/")[-1]
        st.spinner("Cloning the repository...")
        with tempfile.TemporaryDirectory() as local_path:
            if clone_github_repo(github_url, local_path):
                index, documents, file_type_counts, filenames = load_and_index_files(local_path)
                if index is None:
                    st.info("No documents were found to index. Exiting.")
                    exit()

                st.success("Repository cloned. Indexing files...")
                llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0.2)

                template = """
                Repo: {repo_name} ({github_url}) | Conv: {conversation_history} | Docs: {numbered_documents} | Q: {question} | FileCount: {file_type_counts} | FileNames: {filenames}

                Instr:
                1. Answer based on context/docs.
                2. Focus on repo/code.
                3. Consider:
                    a. Purpose/features - describe.
                    b. Functions/code - provide details/samples.
                    c. Setup/usage - give instructions.
                4. Unsure? Say "I am not sure".

                Answer:
                """

                prompt = PromptTemplate(
                    template=template,
                    input_variables=["repo_name", "github_url", "conversation_history", "question", "numbered_documents", "file_type_counts", "filenames"]
                )

                llm_chain = LLMChain(prompt=prompt, llm=llm)

                conversation_history = ""
                question_context = QuestionContext(index, documents, llm_chain, model_name, repo_name, github_url, conversation_history, file_type_counts, filenames)
                while True:
                    try:
                        user_question = st.text_input("Ask a question about the repository (type 'exit()' to quit): ")
                        if user_question.lower() == "exit()":
                            break
                        st.spinner('Thinking...')
                        user_question = format_user_question(user_question)

                        answer = ask_question(user_question, question_context)
                        st.success('ANSWER' + answer)
                        conversation_history += f"Question: {user_question}\nAnswer: {answer}\n"
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        break

            else:
                st.error("Failed to clone the repository.")

if __name__ == "__main__":
    main()