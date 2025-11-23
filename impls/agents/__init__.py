from agents.crl import CRLAgent
from agents.diffuser_diffusion_agent import DiffuserDiffusionAgent
from agents.diffuser_value_agent import DiffuserValueAgent
from agents.flow_bc import FlowBCAgent
from agents.flow_gcbc import FlowGCBCAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.hiql_fm import HIQLFMAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    hiql_fm=HIQLFMAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    diffuser_value=DiffuserValueAgent,
    diffuser_diffusion=DiffuserDiffusionAgent,
    flow_bc=FlowBCAgent,
    flow_gcbc=FlowGCBCAgent,
)
