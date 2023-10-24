try:
    import sage.all
    SAGE_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_AVAILABLE = False
