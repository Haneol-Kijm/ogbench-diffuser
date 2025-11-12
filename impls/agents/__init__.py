from agents.crl import CRLAgent
from agents.diffuser_diffusion_agent import DiffuserDiffusionAgent
from agents.diffuser_value_agent import DiffuserValueAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    diffuser_value=DiffuserValueAgent,
    diffuser_diffusion=DiffuserDiffusionAgent,
)
