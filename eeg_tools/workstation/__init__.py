"""Application-layer models for the desktop EEG workstation."""

from .dataset import DatasetRepository
from .gateway import AcquisitionApp, WorkstationGateway
from .openbci_workspace import OpenBCIWorkspaceManager, OpenBCIWorkspaceState
from .ssvep import SSVEPProtocol, SSVEPProtocolError, SSVEPTrial
from .task import SSVEPTask, TaskPhase, TaskSnapshot

__all__ = [
    "DatasetRepository",
    "AcquisitionApp",
    "SSVEPProtocol",
    "SSVEPProtocolError",
    "SSVEPTask",
    "SSVEPTrial",
    "TaskPhase",
    "TaskSnapshot",
    "WorkstationGateway",
    "OpenBCIWorkspaceManager",
    "OpenBCIWorkspaceState",
]
