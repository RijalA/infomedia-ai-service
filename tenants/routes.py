from fastapi import APIRouter, Request, status
from pydantic import BaseModel, Field
from agents.agentpintar import AgenPintar


route = APIRouter(prefix="/tenants", tags=["tenants"])


class PayloadCreateTenant(BaseModel):
    tenant_name: str = Field(example="Telkom")
    agent_name: str = Field(example="Veronica")


@route.post("/create")
async def create_new_tenant(request: Request, pl: PayloadCreateTenant):
    agents = request.app.state.agents
    tenant_id = pl.tenant_name.upper()
    if tenant_id in agents.keys():
        result = {
            "code": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "status": "Not Created",
            "message": f"Agent with tenant_id {tenant_id} already exist"
        }
        return result
    
    agen = AgenPintar(vendor=tenant_id, agen_name= pl.agent_name)
    request.app.state.agents.update({
        tenant_id: agen
    })
    result = {
        "code": status.HTTP_201_CREATED,
        "status": "Created",
        "message": f"Created agent for {tenant_id}"
    }
    return result


@route.post("/agent")
async def test_agent(request: Request, name: str):
    return {"duar": "mewek"}

