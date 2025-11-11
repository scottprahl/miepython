"""
JupyterLite build configuration for the miepython project.

This script defines the build behavior of JupyterLite when invoked by
`jupyter lite build`. It sets the notebook source directory, output location,
and ensures that the JupyterLab extension paths are properly resolved from
the local virtual environment. In particular, it adds the virtual environment's
`share/jupyter/labextensions` directory so that locally installed federated
extensions (such as the Pyodide kernel) are included in the static site build.
"""

import os

c = get_config()  # noqa: F821
c.LiteBuildApp.apps = ["lab"]

venv_base = os.path.join(os.getcwd(), ".venv")
labext_path = os.path.join(venv_base, "share/jupyter/labextensions")
c.FederatedExtensionAddon.extra_labextensions_path = [labext_path]
