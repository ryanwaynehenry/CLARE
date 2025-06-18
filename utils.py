import os, re, sys
from PyQt5.QtCore import QBuffer, QByteArray, Qt
from PyQt5.QtGui import QMovie, QValidator
from PyQt5.QtWidgets import QLabel, QDoubleSpinBox, QTableWidgetItem

import tiktoken
import string
import time
import traceback
import yaml
from litellm import completion

from docx import Document
from PyPDF2 import PdfReader
import pandas as pd

def resource_path(rel):
    """Return absolute path to a bundled resource."""
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel)

if not os.path.exists("config.yaml"):
    with open("config.yaml", "w") as f:
        yaml.safe_dump({}, f)

# now load it for real
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f) or {}

"""Sentence-Transformers (local) embedding models"""
local_embedding_models = [
    "all-MiniLM-L6-v2",                # very fast / small
    "paraphrase-multilingual-MiniLM-L12-v2",
    "all-mpnet-base-v2"               # high-quality mpnet
]

"""Open AI Models"""
openai_llm_models = [
    "o1-mini",
    "o1-preview",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613"
]
openai_embedding_models = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002"
]

"""Google Models"""
google_llm_models = [
    "gemini/gemini-pro",
    "gemini/gemini-1.5-pro-latest",
    "gemini/gemini-2.0-flash",
    "gemini/gemini-2.0-flash-exp",
    "gemini/gemini-2.0-flash-lite-preview-02-05"
]
google_embedding_models =[
    "gemini/text-embedding-004"
]

"""Anthropic Models"""
anthropic_llm_models = [
    "claude-3-5-sonnet-20240620",
    "claude-3-haiku-20240307",
    "claude-3-opus-20240229",
    "claude-3-5-sonnet-20240620",
    "claude-3-sonnet-20240229",
    "claude-2.1",
    "claude-2",
    "claude-instant-1.2",
    "claude-instant-1"
]

"""AWS Bedrock Models"""
bedrock_llm_models = [
    "bedrock/us.deepseek.r1-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    "bedrock/anthropic.claude-v2:1",
    "bedrock/anthropic.claude-v2",
    "bedrock/anthropic.claude-instant-v1",
    "bedrock/meta.llama3-1-405b-instruct-v1:0",
    "bedrock/meta.llama3-1-70b-instruct-v1:0",
    "bedrock/meta.llama3-1-8b-instruct-v1:0",
    "bedrock/meta.llama3-70b-instruct-v1:0",
    "bedrock/meta.llama3-8b-instruct-v1:0",
    "bedrock/amazon.titan-text-lite-v1",
    "bedrock/amazon.titan-text-express-v1",
    "bedrock/cohere.command-text-v14",
    "bedrock/ai21.j2-mid-v1",
    "bedrock/ai21.j2-ultra-v1",
    "bedrock/ai21.jamba-instruct-v1:0",
    "bedrock/meta.llama2-13b-chat-v1",
    "bedrock/meta.llama2-70b-chat-v1",
    "bedrock/mistral.mistral-7b-instruct-v0:2",
    "bedrock/mistral.mixtral-8x7b-instruct-v0:1"
]
bedrock_embedding_models = [
    "amazon.titan-embed-text-v1",
    "cohere.embed-english-v3",
    "cohere.embed-multilingual-v3"
]

"""Cohere Models"""
cohere_llm_models = [
    "command-r-plus-08-2024",
    "command-r-08-2024",
    "command-r-plus",
    "command-r",
    "command-light",
    "command-nightly"
]
cohere_embedding_models = [
    "embed-english-v3.0",
    "embed-english-light-v3.0",
    "embed-multilingual-v3.0",
    "embed-multilingual-light-v3.0",
    "embed-english-v2.0",
    "embed-english-light-v2.0",
    "embed-multilingual-v2.0"
]

"""Nvidia NIM Models"""
nividia_llm_models = [
    "nvidia_nim/nvidia/nemotron-4-340b-reward",
    "nvidia_nim/01-ai/yi-large",
    "nvidia_nim/aisingapore/sea-lion-7b-instruct",
    "nvidia_nim/databricks/dbrx-instruct",
    "nvidia_nim/google/gemma-7b",
    "nvidia_nim/google/gemma-2b",
    "nvidia_nim/google/codegemma-1.1-7b",
    "nvidia_nim/google/codegemma-7b",
    "nvidia_nim/google/recurrentgemma-2b",
    "nvidia_nim/ibm/granite-34b-code-instruct",
    "nvidia_nim/ibm/granite-8b-code-instruct",
    "nvidia_nim/mediatek/breeze-7b-instruct",
    "nvidia_nim/meta/codellama-70b",
    "nvidia_nim/meta/llama2-70b",
    "nvidia_nim/meta/llama3-8b",
    "nvidia_nim/meta/llama3-70b",
    "nvidia_nim/microsoft/phi-3-medium-4k-instruct",
    "nvidia_nim/microsoft/phi-3-mini-128k-instruct",
    "nvidia_nim/microsoft/phi-3-mini-4k-instruct",
    "nvidia_nim/microsoft/phi-3-small-128k-instruct",
    "nvidia_nim/microsoft/phi-3-small-8k-instruct",
    "nvidia_nim/mistralai/codestral-22b-instruct-v0.1",
    "nvidia_nim/mistralai/mistral-7b-instruct",
    "nvidia_nim/mistralai/mistral-7b-instruct-v0.3",
    "nvidia_nim/mistralai/mixtral-8x7b-instruct",
    "nvidia_nim/mistralai/mixtral-8x22b-instruct",
    "nvidia_nim/mistralai/mistral-large",
    "nvidia_nim/nvidia/nemotron-4-340b-instruct",
    "nvidia_nim/seallms/seallm-7b-v2.5",
    "nvidia_nim/snowflake/arctic",
    "nvidia_nim/upstage/solar-10.7b-instruct"
]
nvidia_embedding_models = [
    "nvidia_nim/NV-Embed-QA",
    "nvidia_nim/nvidia/nv-embed-v1",
    "nvidia_nim/nvidia/nv-embedqa-mistral-7b-v2",
    "nvidia_nim/nvidia/nv-embedqa-e5-v5",
    "nvidia_nim/nvidia/embed-qa-4",
    "nvidia_nim/nvidia/llama-3.2-nv-embedqa-1b-v1",
    "nvidia_nim/nvidia/llama-3.2-nv-embedqa-1b-v2",
    "nvidia_nim/snowflake/arctic-embed-l",
    "nvidia_nim/baai/bge-m3"
]

"""Ollama Models"""
ollama_llm_models =[
    "ollama/mistral",
    "ollama/mistral-7B-Instruct-v0.1",
    "ollama/mistral-7B-Instruct-v0.2",
    "ollama/mistral-8x7B-Instruct-v0.1",
    "ollama/mixtral-8x22B-Instruct-v0.1",
    "ollama/llama2",
    "ollama/llama2:13b",
    "ollama/llama2:70b",
    "ollama/llama2-uncensored",
    "ollama/codellama",
    "ollama/llama3",
    "ollama/llama3:70b",
    "ollama/orca-mini",
    "ollama/vicuna",
    "ollama/nous-hermes",
    "ollama/nous-hermes:13b",
    "ollama/wizard-vicuna"
]
ollama_embedding_models =[
    "mxbai-embed-large",
    "nomic-embed-text",
    "all-minilm",
    "bge-large",
    "paraphrase-multilingual",
    "snowflake-arctic-embed2",
    "granite-embedding"
]

"""Groq Models"""
groq_llm_models = [
    "groq/llama-3.1-8b-instant",
    "groq/llama-3.1-70b-versatile",
    "groq/llama3-8b-8192",
    "groq/llama3-70b-8192",
    "groq/llama2-70b-4096",
    "groq/mixtral-8x7b-32768",
    "groq/gemma-7b-it"
]


def determine_llm_parent(llm_model):
    """Return the model list for the given provider name."""
    if llm_model in openai_llm_models:
        return "openai_llm"
    elif llm_model in google_llm_models:
        return "google_llm"
    elif llm_model in anthropic_llm_models:
        return "anthropic_llm"
    elif llm_model in bedrock_llm_models:
        return "bedrock_llm"
    elif llm_model in cohere_llm_models:
        return "cohere_llm"
    elif llm_model in nividia_llm_models:
        return "nvidia_llm"
    elif llm_model in ollama_llm_models:
        return "ollama_llm"
    elif llm_model in groq_llm_models:
        return "groq_llm"
    else:
        return "ERROR"

def determine_embedding_parent(embedding_model):
    """Return the model list for the given provider name."""
    if embedding_model in openai_embedding_models:
        return "openai_embedding"
    elif embedding_model in google_embedding_models:
        return "google_embedding"
    elif embedding_model in bedrock_embedding_models:
        return "bedrock_embedding"
    elif embedding_model in cohere_embedding_models:
        return "cohere_embedding"
    elif embedding_model in nvidia_embedding_models:
        return "nvidia_embedding"
    elif embedding_model in ollama_embedding_models:
        return "ollama_embedding"
    elif embedding_model in local_embedding_models:
        return "local_embedding"
    else:
        return "ERROR"

def set_env_variables(llm_model: str, embedding_model: str, llm_keys: list, embedding_keys: list):
    llm_set = False
    embedding_set = False
    llm_parent = determine_llm_parent(llm_model)
    if llm_parent == "openai_llm":
        os.environ["OPENAI_API_KEY"] = llm_keys[0]
        llm_set = True
    elif llm_parent == "google_llm":
        os.environ["GEMINI_API_KEY"] = llm_keys[0]
        llm_set = True
    elif llm_parent == "anthropic_llm":
        os.environ["ANTHROPIC_API_KEY"] = llm_keys[0]
        llm_set = True
    elif llm_parent == "bedrock_llm":
        os.environ["AWS_ACCESS_KEY_ID"] = llm_keys[0]
        os.environ["AWS_SECRET_ACCESS_KEY"] = llm_keys[1]
        os.environ["AWS_REGION_NAME"] = llm_keys[2]
        llm_set = True
    elif llm_parent == "cohere_llm":
        os.environ["COHERE_API_KEY"] = llm_keys[0]
        llm_set = True
    elif llm_parent == "nvidia_llm":
        os.environ["NVIDIA_NIM_API_KEY"] = llm_keys[0]
        llm_set = True
    elif llm_parent == "ollama_llm":
        llm_set = True
    elif llm_parent == "groq_llm":
        os.environ["GROQ_API_KEY"] = llm_keys[0]
        llm_set = True

    embedding_parent = determine_embedding_parent(embedding_model)
    if embedding_parent == "openai_embedding":
        os.environ["OPENAI_API_KEY"] = embedding_keys[0]
        embedding_set = True
    elif embedding_parent == "google_embedding":
        os.environ["GEMINI_API_KEY"] = embedding_keys[0]
        embedding_set = True
    elif embedding_parent == "bedrock_embedding":
        os.environ["AWS_ACCESS_KEY_ID"] = embedding_keys[0]
        os.environ["AWS_SECRET_ACCESS_KEY"] = embedding_keys[1]
        os.environ["AWS_REGION_NAME"] = embedding_keys[2]
        embedding_set = True
    elif embedding_parent == "cohere_embedding":
        os.environ["COHERE_API_KEY"] = embedding_keys[0]
        embedding_set = True
    elif embedding_parent == "nvidia_embedding":
        os.environ["NVIDIA_NIM_API_KEY"] = embedding_keys[0]
        embedding_set = True
    elif embedding_parent == "ollama_embedding":
        embedding_set = True
    elif embedding_parent == "local_embedding":
        embedding_set = True

    return llm_set, embedding_set


def get_num_tokens(texts, model):
    num_tokens = []
    encoding = tiktoken.encoding_for_model(model)
    for text in texts:
        num_tokens.append(len(encoding.encode(text)))
    return num_tokens

def get_completion(prompt, model_name="gpt-4o", max_tokens=2048, retry_times=3, temperature=0.3, top_p=1.0, llm_api_key=""):
    """
    Generate chat completions using litellm.

    Parameters:
    - prompt: The input prompt for the chat.
    - model_name: The model to use for generating completions. Default is "gpt-4o".
    - temperature: Controls randomness. Lower values make the model more deterministic. Default is 0.3.
    - max_tokens: The maximum number of tokens to generate. Default is 2048.
    - top_p: Controls diversity via nucleus sampling. Default is 1.0.
    - retry_times: Number of retries if the request fails. Default is 3.
    - llm_api_key: The API base (or key) to use for models that require it (e.g. Ollama).

    Returns:
    A tuple containing:
      - response: the text generated by the model,
      - time_taken: the duration of the request,
      - tokens_used: the number of tokens consumed.
    """
    for attempt in range(retry_times):
        try:
            start_time = time.time()
            # Check if the model is from Ollama and add the api_base parameter if it is.
            if determine_llm_parent(model_name) == "ollama_llm":
                chat_completion = completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    api_base=llm_api_key
                )
            else:
                chat_completion = completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )

            end_time = time.time()
            response = chat_completion.choices[0].message.content
            time_taken = end_time - start_time
            tokens_used = chat_completion.usage.total_tokens

            return response, time_taken, tokens_used

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(30)  # Wait for 30 seconds before retrying

    raise ValueError("Failed to generate completion after retrying.")

def process_strings(strings):
    def normalize(s):
        return ''.join(c.lower() for c in s if c not in string.whitespace and c not in string.punctuation)

    strings = [s.strip() for s in strings]

    unique_strings = []
    normalized_set = set()

    for s in strings:
        normalized_s = normalize(s)
        if normalized_s not in normalized_set:
            normalized_set.add(normalized_s)
            unique_strings.append(s)

    return unique_strings

def remove_duplicates(S):
    dic = {}
    for i, j in S:
        if i not in dic:
            dic[i] = [j]
        else:
            if j not in dic[i]:
                dic[i].append(j)

    no_duplicates = []
    for i, j in dic.items():
        no_duplicates.append((i, j))

    return no_duplicates

def clean_text(text):
    text = text.replace("///", " ").replace("_x000D_", " ").replace("、", " ")

    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'\n\n+', '\n\n', text)

    text = text.strip()

    return text

def my_text_splitter(input_text, chunk_size=200, separator=None, model='gpt-4o'):
    if separator is None:
        separator = ["\n\n", ".", " "]

    if not separator:
        return []
    outputs = []
    sep = separator[0]
    seg_texts = input_text.split(sep)
    text_tokens = get_num_tokens(seg_texts, model)
    temp_tokens = 0
    for text, text_token in zip(seg_texts, text_tokens):
        if not text:
            continue
        if temp_tokens + text_token <= chunk_size:
            if temp_tokens == 0:
                outputs.append(text)
            else:
                outputs[-1] += sep + text
            temp_tokens += text_token
        else:
            if text_token > chunk_size:
                sub_output = my_text_splitter(text, chunk_size, separator[1:])
                outputs.extend(sub_output)
                temp_tokens = 0
            else:
                outputs.append(text)
                temp_tokens = text_token

    return outputs


def split_texts_with_source(text_list, source_list, chunk_size=200, separator=None):
    seg_texts = []
    new_sources = []
    for i, text in enumerate(text_list):
        try:
            temp_seg_texts = my_text_splitter(text, chunk_size=chunk_size, separator=separator)
            seg_texts.extend(temp_seg_texts)
            new_sources.extend([source_list[i] for _ in temp_seg_texts])
        except:
            raise ValueError(f"{i}" + traceback.format_exc())

    return seg_texts, new_sources

def find_files(directory, filetype='docx'):
    docx_files = []
    sub_paths = []

    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(f".{filetype}"):
                docx_files.append(filename)
                sub_paths.append(dirpath)

    return docx_files, sub_paths

def load_and_process_files(directory,
                           chunk_size=200,
                           separator=None):
    '''
    Load all docx, pdf, and xlsx files under the directory
    Then segment them into text chunks
    '''
    if separator is None:
        separator = ["\n\n", "\n", ". ", " "]

    raw_texts = []
    raw_sources = []

    # docx
    docx_files, sub_paths = find_files(directory, filetype='docx')

    for docx_file, sub_path in zip(docx_files, sub_paths):
        try:
            doc = Document(os.path.join(sub_path, docx_file))
            raw_texts.append("\n".join([t.text for t in doc.paragraphs]))
            raw_sources.append(docx_file)
        except Exception as e:
            print("Failed to load the file:", sub_path, docx_file)
            print("Error message:", str(e))
            continue

    # pdf
    pdf_files, sub_paths = find_files(directory, filetype='pdf')

    for pdf_file, sub_path in zip(pdf_files, sub_paths):
        raw_sources.append(pdf_file)
        try:
            pdf_path = os.path.join(sub_path, pdf_file)
            with open(pdf_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()  # 使用新方法 extract_text
                raw_texts.append(text)
        except Exception as e:
            print("Failed to load the file:", sub_path, pdf_file)
            print("Error message:", str(e))
            continue

    # xlsx
    xlsx_files, sub_paths = find_files(directory, filetype='xlsx')

    for xlsx_file, sub_path in zip(xlsx_files, sub_paths):
        try:
            xlsx_path = os.path.join(sub_path, xlsx_file)
            df = pd.read_excel(xlsx_path)

            if 'text' in df.columns:
                raw_texts.extend(df['text'].tolist())

            if 'source' in df.columns:
                source_list = df['source'].tolist()
                raw_sources.extend([xlsx_file + '\n' + s for s in source_list])


        except Exception as e:
            print("Failed to load the file:", sub_path, xlsx_file)
            print("Error message:", str(e))
            continue

    # clean text
    processed_texts = [clean_text(text) for text in raw_texts]

    # segmentation
    texts, sources = split_texts_with_source(processed_texts,
                                             raw_sources,
                                             chunk_size=chunk_size,
                                             separator=separator)

    return texts, sources

# Spinner GIF data
spinner_gif_data = b'R0lGODlhEAAQAPIAAP///wAAAMLCwkJCQgAAAAAAACH5BAEAAAIALAAAAAAQABAAAAM5SLrc/jDKSau9OOvNu/9gKI5kaZ5oqubL7D0b00Zp3f37gQA7'

def create_spinner():
    spinner_label = QLabel()
    spinner_label.setFixedSize(24, 24)
    movie = QMovie()
    buffer = QBuffer()
    buffer.setData(QByteArray(spinner_gif_data))
    buffer.open(QBuffer.ReadOnly)
    movie.setDevice(buffer)
    spinner_label.setMovie(movie)
    movie.start()
    return spinner_label

def seconds_to_formatted(seconds):
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds - minutes * 60
        return f"{minutes}:{secs:04.1f}"
    else:
        hours = int(seconds // 3600)
        remainder = seconds % 3600
        minutes = int(remainder // 60)
        secs = remainder - minutes * 60
        return f"{hours}:{minutes:02d}:{secs:04.1f}"

def formatted_to_seconds(time_str):
    parts = time_str.split(":")
    if len(parts) == 2:
        m = int(parts[0])
        s = float(parts[1])
        return m * 60 + s
    elif len(parts) == 3:
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s
    else:
        raise ValueError("Invalid time format")

class TimeSpinBox(QDoubleSpinBox):
    """
    A custom spin box that displays and accepts time in m:ss.d or h:mm:ss.d format.
    The user may type anything; the value (in seconds) is only updated when the user
    clicks outside the box or presses Enter. If the entered text is invalid, it reverts
    to the last valid value.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # Disable keyboard tracking so that value isn't updated on every key press.
        self.setKeyboardTracking(False)
        self.lineEdit().setValidator(None)
        # Store the last valid value (in seconds)
        self._lastValidValue = 0.0
        # Optionally, set a default value:
        self.setValue(0.0)

    def textFromValue(self, value):
        # If less than one hour, format as m:ss.d; otherwise, h:mm:ss.d.
        if value < 3600:
            minutes = int(value // 60)
            secs = value - minutes * 60
            return f"{minutes}:{secs:04.1f}"
        else:
            hours = int(value // 3600)
            remainder = value % 3600
            minutes = int(remainder // 60)
            secs = remainder - minutes * 60
            return f"{hours}:{minutes:02d}:{secs:04.1f}"

    def focusOutEvent(self, event):
        # When focus is lost, validate the text.
        self.validate_and_update()
        super().focusOutEvent(event)

    def keyPressEvent(self, event):
        # If the user presses Enter, validate and update.
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.validate_and_update()
        else:
            super().keyPressEvent(event)

    def validate_and_update(self):
        text = self.lineEdit().text().strip()
        # Only update if there is something typed.
        if text:
            try:
                new_val = formatted_to_seconds(text)
                self.setValue(new_val)
                self._lastValidValue = new_val
            except Exception:
                # If invalid, revert to the last valid value.
                self.lineEdit().setText(self.textFromValue(self._lastValidValue))
        else:
            # If the field is empty, revert to the last valid text.
            self.lineEdit().setText(self.textFromValue(self._lastValidValue))
class TimeTableWidgetItem(QTableWidgetItem):
    def __lt__(self, other):
        try:
            self_time = formatted_to_seconds(self.text())
        except Exception:
            self_time = 0
        try:
            other_time = formatted_to_seconds(other.text())
        except Exception:
            other_time = 0
        return self_time < other_time
