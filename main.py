import os
import sys
import traceback
from dotenv import load_dotenv
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QLineEdit,
    QTabWidget, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox
)
import asyncio
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ---- AGENT SYSTEM ----
from agents import (
    Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,
    handoff, set_tracing_disabled, function_tool
)
from tavily import AsyncTavilyClient

# Load environment variables
load_dotenv()

# ✅ STEP 1: Load all Gemini + Tavily keys from .env
GEMINI_KEYS = [os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 11)]
TAVILY_KEYS = [os.getenv(f"TAVILY_API_KEY_{i}") for i in range(1, 11)]

# ---- API Key Manager Class (ADDED BY GPT) ----
class APIKeyManager:
    def __init__(self, keys, log_callback=None):
        self.keys = [k for k in keys if k]  # remove None keys
        self.index = 0
        self.log_callback = log_callback

    def get_key(self):
        if not self.keys:
            raise ValueError("No API keys available.")
        key = self.keys[self.index]
        if self.log_callback:
            self.log_callback(f"🔑 Using API Key No. {self.index + 1}")
        return key

    def rotate_key(self):
        self.index = (self.index + 1) % len(self.keys)
        key = self.keys[self.index]
        if self.log_callback:
            self.log_callback(f"🔄 Switching to API Key No. {self.index + 1}")
        return key


# ✅ STEP 2: Initialize managers
gemini_manager = APIKeyManager(GEMINI_KEYS)
tavily_manager = APIKeyManager(TAVILY_KEYS)

set_tracing_disabled(disabled=False)

# Clients function (ADDED BY GPT: dynamically rebuilds clients when key rotates)
def get_clients():
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_key = gemini_manager.get_key()
    tavily_key = tavily_manager.get_key()
    external_client = AsyncOpenAI(api_key=gemini_key, base_url=BASE_URL)
    tavily_client = AsyncTavilyClient(api_key=tavily_key)
    return external_client, tavily_client

external_client, tavily_client = get_clients()

# Models
flashlite = OpenAIChatCompletionsModel(model="gemini-2.5-flash-lite", openai_client=external_client)
flash = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=external_client)

# Tools
@function_tool()
async def search(query: str) -> str:
    return await tavily_client.search(query, max_results=2)

@function_tool()
async def extract_context(urls: list) -> dict:
    return await tavily_client.extract_context(urls)

# Agents
planning_agent = Agent(
    name="PlanningAgent",
    model=flashlite,
    instructions="Break down the user query into clear, factual steps.",
)

parallel_research_agent = Agent(
    name="ParallelResearchAgent",
    model=flash,
    tools=[search, extract_context],
    instructions="Research steps using search + context tools. Return facts.",
)

summarizer_agent = Agent(
    name="SummarizerAgent",
    model=flashlite,
    instructions="Summarize clearly using bullet points, avoid tables, stats.",
)

# Tools
planning_tool = planning_agent.as_tool("planning_tool", "Break question into steps.")
parallel_tool = parallel_research_agent.as_tool("parallel_research_tool", "Run facts in parallel.")

# Controller
agent = Agent(
    name="SearchAgent",
    model=flash,
    tools=[planning_tool, parallel_tool],
    handoffs=[handoff(summarizer_agent)],
    instructions="Use planning → parallel → summarize → return clean output.",
)

# Threaded runner with logs
class AgentRunnerThread(QThread):
    result_ready = pyqtSignal(str)
    log_ready = pyqtSignal(str)

    def __init__(self, query):
        super().__init__()
        self.query = query

    def log(self, msg):
        self.log_ready.emit(msg)

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # ✅ STEP 3: Show which keys are being used
            gemini_manager.log_callback = self.log
            tavily_manager.log_callback = self.log
            get_clients()

            self.log("📝 Planning the steps...")
            loop.run_until_complete(asyncio.sleep(1))

            self.log("🔎 Searching information...")
            loop.run_until_complete(asyncio.sleep(1))

            self.log("📚 Extracting context...")
            loop.run_until_complete(asyncio.sleep(1))

            self.log("🧠 Summarizing findings...")
            loop.run_until_complete(asyncio.sleep(1))

            result = loop.run_until_complete(Runner.run(agent, self.query))
            self.result_ready.emit(result.final_output)
        except Exception as e:
            # ✅ STEP 4: If key fails, rotate to next one
            self.log("⚠️ Error occurred, rotating API key...")
            gemini_manager.rotate_key()
            tavily_manager.rotate_key()
            get_clients()
            self.result_ready.emit(f"❌ Error: {str(e)}\n\n{traceback.format_exc()}")

# -------- GUI --------
class AgentApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemini Research Assistant")
        self.setMinimumSize(1100, 750)
        self.setFont(QFont("Segoe UI", 11))
        self.init_ui()
        self.set_styles()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(30, 20, 30, 20)

        title = QLabel("🔍 Gemini AI Research Assistant")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Segoe UI", 10))

        # --- Main tab ---
        main_tab = QWidget()
        main_layout = QVBoxLayout()

        input_layout = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("What do you want to research today?")
        self.input_box.setMinimumHeight(45)
        input_layout.addWidget(self.input_box)

        self.run_button = QPushButton("Run")
        self.run_button.setMinimumHeight(45)
        self.run_button.clicked.connect(self.handle_run)
        input_layout.addWidget(self.run_button)
        main_layout.addLayout(input_layout)

        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setFont(QFont("Consolas", 12))
        self.output_area.setLineWrapMode(QTextEdit.WidgetWidth)
        main_layout.addWidget(self.output_area)
        main_tab.setLayout(main_layout)
        self.tabs.addTab(main_tab, "Main")

        # --- Thinking tab ---
        self.thinking_tab = QWidget()
        thinking_layout = QVBoxLayout()
        self.thinking_logs = QTextEdit()
        self.thinking_logs.setReadOnly(True)
        self.thinking_logs.setFont(QFont("Consolas", 11))
        self.thinking_logs.setLineWrapMode(QTextEdit.WidgetWidth)
        thinking_layout.addWidget(self.thinking_logs)
        self.thinking_tab.setLayout(thinking_layout)
        self.tabs.addTab(self.thinking_tab, "Thinking")

        # --- Settings tab ---
        settings_tab = QWidget()
        form_layout = QFormLayout()
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(100, 8000)
        self.max_tokens.setValue(4999)
        form_layout.addRow("Max Tokens:", self.max_tokens)

        self.model_box = QComboBox()
        self.model_box.addItems(["gemini-2.5-flash", "gemini-2.5-flash-lite"])
        self.model_box.setCurrentIndex(0)
        form_layout.addRow("Model:", self.model_box)

        self.provider_box = QComboBox()
        self.provider_box.addItems(["Google Gemini", "OpenAI", "Anthropic"])
        self.provider_box.setCurrentIndex(0)
        form_layout.addRow("Provider:", self.provider_box)

        self.temp_box = QDoubleSpinBox()
        self.temp_box.setRange(0.0, 1.0)
        self.temp_box.setSingleStep(0.1)
        self.temp_box.setValue(0.7)
        form_layout.addRow("Temperature:", self.temp_box)
        settings_tab.setLayout(form_layout)
        self.tabs.addTab(settings_tab, "Settings")

        # --- Personal tab ---
        personal_tab = QWidget()
        personal_layout = QFormLayout()
        self.name_input = QLineEdit()
        personal_layout.addRow("Your Name:", self.name_input)
        self.job_input = QLineEdit()
        personal_layout.addRow("Your Job:", self.job_input)
        self.desc_input = QTextEdit()
        self.desc_input.setLineWrapMode(QTextEdit.WidgetWidth)
        personal_layout.addRow("Description:", self.desc_input)
        self.age_input = QLineEdit()
        personal_layout.addRow("Your Age:", self.age_input)
        personal_tab.setLayout(personal_layout)
        self.tabs.addTab(personal_tab, "Personal")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def handle_run(self):
        query = self.input_box.text().strip()
        if not query:
            self.output_area.setText("⚠️ Please enter a valid question.")
            return
        self.output_area.setText("⏳ Running agents... Please wait.")
        self.thinking_logs.setText("")
        self.thread = AgentRunnerThread(query)
        self.thread.result_ready.connect(self.display_result)
        self.thread.log_ready.connect(self.update_logs)
        self.thread.start()

    def display_result(self, text):
        self.output_area.setPlainText(text)

    def update_logs(self, log):
        self.thinking_logs.append(log)

    def set_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                color: #111111;
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel#titleLabel {
                font-size: 28px;
                font-weight: bold;
                margin-bottom: 15px;
            }
            QLineEdit, QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #cccccc;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #333333;
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                padding: 0 20px;
            }
            QPushButton:hover {
                background-color: #ff9900;
                color: #000000;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 6px;
                margin-top: -1px;
            }
            QTabBar::tab {
                background: #eaeaea;
                padding: 10px 20px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                border-bottom: 2px solid #ff9900;
            }
        """)

# -------- APP RUN --------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AgentApp()
    window.show()
    sys.exit(app.exec_())
