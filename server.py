from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agents import build_graph

app = FastAPI()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the LangGraph agent
agent_app = build_graph()

class ResearchRequest(BaseModel):
    query: str

class ResearchResponse(BaseModel):
    report: str

@app.post("/api/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    try:
        state = {"messages": [HumanMessage(content=request.query)]}
        result = agent_app.invoke(state)
        return {"report": result.get("final_report", "Error generating report.")}
    except Exception as e:
        return {"report": f"An error occurred: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
