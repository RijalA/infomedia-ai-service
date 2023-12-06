from fastapi import FastAPI
from agents.routes import route as AgentRoutes
from knowledgebase.routes import routes as KBRoutes
from tenants.routes import route as TenantsRoutes

app = FastAPI(
    title="LangChain"
)

app.state.agents = {
    "test": "TEST"
}

app.include_router(AgentRoutes)
app.include_router(KBRoutes)
app.include_router(TenantsRoutes)


