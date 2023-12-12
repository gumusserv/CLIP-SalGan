# Reading and processing all four JSON files to prepare for the combined subplots
import json
import matplotlib.pyplot as plt

# File paths for the four JSON files
files = ['loss_total.json', 'loss_general.json', 
         'loss_nonsal.json', 'loss_sal.json']

# Function to extract loss data from a JSON file
def extract_losses(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    step_g_losses, step_d_losses, val_losses, step_counts = [], [], [], []
    step_counter = 0
    epochs = list(data.keys())

    for epoch in epochs:
        for step in data[epoch]:
            if step != 'Final':
                step_g_losses.append(data[epoch][step]['G LOSS'])
                step_d_losses.append(data[epoch][step]['D LOSS'])
                step_counts.append(step_counter)
                step_counter += 1
            else:
                step_g_losses.append(data[epoch][step]['Train G Loss'])
                step_d_losses.append(data[epoch][step]['Train D Loss'])
                step_counts.append(step_counter)
                val_losses.append(data[epoch][step]['Val Loss'])
    
    return step_g_losses, step_d_losses, val_losses, step_counts

# Extracting data from all files
loss_data = [extract_losses(file) for file in files]

# Plotting the data in subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
axs = axs.ravel()

for i, (step_g_losses, step_d_losses, val_losses, step_counts) in enumerate(loss_data):
    axs[i].plot(step_counts, step_g_losses, label='G Loss', color='blue')
    axs[i].plot(step_counts, step_d_losses, label='D Loss', color='red')
    axs[i].plot([x for x in range(0, len(val_losses))], val_losses, label='Val Loss', color='green', linestyle='--')
    axs[i].set_xlim(0, 50)
    axs[i].set_xlabel('Epoch')
    axs[i].set_ylabel('Loss')
    if i == 0:
        axs[i].set_title(f'Losses for Model on Total Dataset')
    elif i == 1:
        axs[i].set_title(f'Losses for Model on General Dataset')
    elif i == 2:
        axs[i].set_title(f'Losses for Model on Non-Salient Dataset')
    else:
        axs[i].set_title(f'Losses for Model on Salient Dataset')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()


