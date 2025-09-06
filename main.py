import os
import sys
import traceback
from dotenv import load_dotenv
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QLineEdit, QSizePolicy
)
import asyncio
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ---- AGENT SYSTEM ----
from agents import (
    Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,
    handoff, set_tracing_disabled, function_tool, ModelSettings
)
from tavily import AsyncTavilyClient

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

set_tracing_disabled(disabled=False)

# Clients
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
external_client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)

# Models
flashlite = OpenAIChatCompletionsModel(model="gemini-2.5-flash-lite", openai_client=external_client)
flash = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=external_client)

# Tools
@function_tool()
async def search(query: str) -> str:
    print(f"🔎 Searching: {query}")
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
    instructions="Summarize clearly using bullet points, tables, stats.",
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
    model_settings=ModelSettings(
        temperature=0.3,
        max_retries=5,
        timeout=90
    ),
)


# Threaded runner
class AgentRunnerThread(QThread):
    result_ready = pyqtSignal(str)

    def __init__(self, query):
        super().__init__()
        self.query = query

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(Runner.run(agent, self.query))
            self.result_ready.emit(result.final_output)
        except Exception as e:
            self.result_ready.emit(f"❌ Error: {str(e)}\n\n{traceback.format_exc()}")


# -------- GUI --------
class AgentApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemini Research Assistant")
        self.setMinimumSize(1024, 700)
        self.init_ui()
        self.set_styles()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(40, 30, 40, 30)

        # Title
        title = QLabel("🔍 Gemini AI Research Assistant")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Input row
        input_layout = QHBoxLayout()
        input_layout.setSpacing(12)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("What do you want to research today?")
        self.input_box.setMinimumHeight(50)
        self.input_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        input_layout.addWidget(self.input_box)

        self.run_button = QPushButton("Run")
        self.run_button.setMinimumHeight(50)
        self.run_button.setCursor(Qt.PointingHandCursor)
        self.run_button.clicked.connect(self.handle_run)
        input_layout.addWidget(self.run_button)

        layout.addLayout(input_layout)

        # Output
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setFont(QFont("Consolas", 12))
        self.output_area.setObjectName("outputBox")
        layout.addWidget(self.output_area)

        self.setLayout(layout)

    def handle_run(self):
        query = self.input_box.text().strip()
        if not query:
            self.output_area.setText("⚠️ Please enter a valid question.")
            return
        self.output_area.setText("⏳ Running agents... Please wait.")
        self.thread = AgentRunnerThread(query)
        self.thread.result_ready.connect(self.display_result)
        self.thread.start()

    def display_result(self, text):
        escaped_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.output_area.setHtml(f"<pre style='color: #000000;'>{escaped_text}</pre>")

    def set_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                color: #000000;
                font-family: 'Segoe UI', sans-serif;
                font-size: 15px;
            }
            QLabel#titleLabel {
                font-size: 26px;
                font-weight: bold;
                color: #000000;
                margin-bottom: 20px;
            }
            QLineEdit {
                background-color: #f9f9f9;
                border: 2px solid #000000;
                border-radius: 10px;
                padding: 14px 16px;
                font-size: 16px;
                color: #000000;
            }
            QLineEdit:focus {
                border-color: #FFA500;
                background-color: #fffefc;
            }
            QPushButton {
                background-color: #000000;
                color: #ffffff;
                font-size: 16px;
                font-weight: bold;
                border: none;
                border-radius: 10px;
                padding: 0 24px;
            }
            QPushButton:hover {
                background-color: #FFA500;
                color: #000000;
            }
            QTextEdit#outputBox {
                background-color: #f3f3f3;
                color: #000000;
                border: 1px solid #cccccc;
                border-radius: 12px;
                padding: 20px;
                font-family: Consolas, monospace;
                font-size: 14px;
                line-height: 1.6;
            }
        """)


# -------- APP RUN --------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AgentApp()
    window.show()
    sys.exit(app.exec_())
