from fastapi import APIRouter, status, Request

from pydantic import BaseModel, Field

route = APIRouter(prefix="/agents", tags=["agent"])


### Payloads
class PayloadAskAgent(BaseModel):
    tenant_id: str = Field(example="Telkom")
    query: str = Field(example="Siapa kah dirimu wahai pujangga?")


@route.post("/ask")
async def ask_agent(request: Request, req: PayloadAskAgent):
    tenant_id = req.tenant_id.upper()
    AGENTS = request.app.state.agents
    if tenant_id not in AGENTS:
        return {
            "status": status.HTTP_400_BAD_REQUEST,
            "error": "Tenants Not Found"
        }
    else:
        agent = AGENTS[tenant_id]
        
    res = agent(req.query)

    result = {
        "status": status.HTTP_200_OK,
        "query": req.query,
        "answer": res
    }

    return result