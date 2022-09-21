def split_data(data, vocs):
    variable_data = vocs.variable_data(data, "")
    objective_data = vocs.objective_data(data, "")
    constraint_data = vocs.constraint_data(data, "")
    return variable_data, objective_data, constraint_data
