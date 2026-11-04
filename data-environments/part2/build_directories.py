import os, getpass

task = 'part2'

# Local/Virtual Machine home
home = f'/home/{getpass.getuser()}'

# Paths
task_dir = os.path.join(home, task)

data_dir = os.path.join(task_dir, 'data')
docs_dir = os.path.join(task_dir, 'docs')
results_dir = os.path.join(task_dir, 'results')
stages_dir = os.path.join(task_dir, 'stages')

raw_dir = os.path.join(data_dir, 'raw')
staged_dir = os.path.join(data_dir, 'staged')
cleaned_dir = os.path.join(data_dir, 'cleaned')
transformed_dir = os.path.join(data_dir, 'transformed')

# Create main project folder if needed
if task not in os.listdir(home):
    os.mkdir(task_dir)

# Create required folders
dirs = [
    data_dir,
    docs_dir,
    results_dir,
    stages_dir,
    raw_dir,
    staged_dir,
    cleaned_dir,
    transformed_dir
]

for directory in dirs:
    if not os.path.exists(directory):
        os.mkdir(directory)
        print(f"Created: {directory}")
    else:
        print(f"Already exists: {directory}")