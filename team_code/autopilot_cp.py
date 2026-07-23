"""AutoPilot variant that uses ``config_cp.GlobalConfig``.

The driving implementation remains in the original ``autopilot.py``. This
small adapter only replaces the configuration factory during setup, keeping
the original source file untouched.
"""

import autopilot as base_autopilot

from config_cp import GlobalConfig


def get_entry_point():
  return "AutoPilot"


class AutoPilot(base_autopilot.AutoPilot):
  """Original privileged expert configured through ``config_cp.py``."""

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    # AutoPilot.setup() constructs GlobalConfig internally. Temporarily replace
    # that module-level factory so all controllers and loggers receive the CP
    # configuration from the beginning, then restore the original binding.
    original_config_factory = base_autopilot.GlobalConfig
    base_autopilot.GlobalConfig = GlobalConfig
    try:
      return super().setup(path_to_conf_file, route_index, traffic_manager)
    finally:
      base_autopilot.GlobalConfig = original_config_factory

