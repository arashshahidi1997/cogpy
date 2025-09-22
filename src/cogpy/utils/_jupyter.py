import dill

def dump_sesion(filename, all_vars, exclude_vars=[]):
    # Get a dictionary of all variables in the current scope
    # this function is called from other modules, so we need to get all variables in the other modules
    # https://stackoverflow.com/questions/8658043/how-to-export-all-the-variables-in-the-module-scope-using-pickle
    # all_vars = globals().copy()
    # all_vars.update(locals())
    
    # Serialize variables one by one, skipping errors
    serialized_vars = {}
    for var_name, var_value in all_vars.items():
        if var_name not in exclude_vars:
            try:
                serialized_vars[var_name] = dill.dumps(var_value)
                print('Serialized', var_name)
            except Exception as e:
                print(f"Error serializing {var_name}: {e}")
    
    # Save the serialized variables to the specified file
    with open(filename, 'wb') as f:
        dill.dump(serialized_vars, f)

def load_session(filename):
    # Load serialized session variables from the specified file
    with open(filename, 'rb') as f:
        serialized_vars = dill.load(f)
    
    var_dict = {}

    # Restore the session variables in the current scope
    for var_name, serialized_value in serialized_vars.items():
        try:
            value = dill.loads(serialized_value)
            var_dict[var_name] = value
        except Exception as e:
            print(f"Error loading {var_name}: {e}")

    return var_dict
