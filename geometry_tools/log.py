VERBOSITY = "silent"

def log(message, level="debug"):
    if level == VERBOSITY:
        print(message)
