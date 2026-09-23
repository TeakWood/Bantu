"""Supervised fleets: several nanobot instances, each in its own OS process."""

from nanobot.fleet.config import Fleet, FleetConfigError, FleetInstance, load_fleet
from nanobot.fleet.environment import BASE_ENV_VARS, instance_environment
from nanobot.fleet.supervisor import Supervisor, stop_fleet

__all__ = [
    "BASE_ENV_VARS",
    "Fleet",
    "FleetConfigError",
    "FleetInstance",
    "Supervisor",
    "instance_environment",
    "load_fleet",
    "stop_fleet",
]
