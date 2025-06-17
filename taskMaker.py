import os
import json

# --- CONFIG --- #
data_dir = "/media/jag/volD/cifer100/cifer/train"  # change to val/ if needed
output_path = "tasks.json"
num_tasks = 4
classes_per_task = 20

# --- READ & SORT CLASS FOLDERS --- #
all_classes = sorted(
    [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))],
    key=lambda x: int(x.split("_", 1)[0])
)

assert len(all_classes) == 100, f"Expected 100 class folders, found {len(all_classes)}."

# --- BUILD TASK DICTIONARY --- #
remaining = all_classes.copy()
already_forgotten = []
tasks = {}

for step in range(1, num_tasks + 1):
    forget = remaining[:classes_per_task]
    retained = remaining[classes_per_task:]

    tasks[f"Step{step}"] = {
        "forget": forget,
        "retained": retained,
        "already_forgotten": already_forgotten.copy()
    }

    # Prepare for next iteration
    already_forgotten += forget
    remaining = retained

# --- SAVE TASKS TO JSON --- #
with open(output_path, "w") as f:
    json.dump(tasks, f, indent=2)

print(f"✅ Generated '{output_path}' with {num_tasks} tasks ({classes_per_task} classes forgotten each).")
