# 🔎 Multi-Agent Research & Summarization System

This project is a **multi-agent research system** that leverages planning, parallel research, and summarization to deliver **clear, factual, and easily scannable answers**.  
It combines **Gemini models** with the **Tavily search API** for up-to-date research.

---

## 🚀 Setup and Usage

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo>
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
# Activate
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory with the following content:
```env
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
TAVILY_API_KEY=your-tavily-key
```

### 5. Start the Agent
```bash
uv run main.py
```

---

## 💡 Example Research Prompts

You can ask questions such as:

- Compare electric vs gas cars
- Pros and cons of nuclear vs solar energy
- Trends in smartphone adoption since 2010
- Health benefits and risks of intermittent fasting
- AI in education: challenges and opportunities

---

## 🧩 Agent Responsibilities

### 1. PlanningAgent 🗂️
- Breaks down queries into clear, numbered research steps
- Covers aspects like costs, performance, benefits, risks, timelines, and statistics
- Generates actionable search queries

### 2. ParallelResearchAgent 🌍
- Executes search queries in parallel using the Tavily API
- Tools:
    - `search` for quick factual lookups
    - `extract_context` for deeper content extraction
- Produces neutral, fact-based mini-summaries

### 3. SummarizerAgent 📝
- Converts research findings into a clean, structured summary
- Uses section headers (e.g., Costs, Performance, Risks)
- Presents bullet points (≤ 12 words each)
- Includes comparison tables when relevant
- Removes unnecessary details and opinions

### 4. SearchAgent (Main Controller) 🎯
- Manages the entire workflow:
    - Sends queries to PlanningAgent
    - Passes research steps to ParallelResearchAgent
    - Forwards results to SummarizerAgent
- Ensures the final output is structured, factual, and comparative

---

## 🔗 Team Workflow

1. **User Query** → PlanningAgent creates a research plan  
2. **Plan** → ParallelResearchAgent gathers facts in parallel  
3. **Findings** → SummarizerAgent formats a structured summary  
4. **Final Answer** → User receives a scannable, factual response

