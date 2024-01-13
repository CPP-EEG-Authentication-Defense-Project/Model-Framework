import invoke


@invoke.task
def requirements(c):
    """
    Re-builds the development requirements file.
    """
    c.run('pip-compile --extra=dev --output-file=dev-requirements.txt pyproject.toml')


@invoke.task
def build(c):
    """
    Locally builds and installs the current development package version.
    """
    c.run('pip install -e .[dev]')


@invoke.task
def test(c):
    """
    Builds the project and tests it.
    """
    c.run('python -m unittest discover -s tests -p test*.py')
