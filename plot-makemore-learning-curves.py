import csv
import matplotlib.pyplot as plt

# Function to read data from CSV file
def read_csv_data(filename):
    step, train_loss, test_loss = [], [], []
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            hex_tag, step_num, train, test = row
            step.append(int(step_num))
            train_loss.append(float(train))
            test_loss.append(float(test))
    return step, train_loss, test_loss

# Provide the path to your CSV file
csv_filename = 'makemore_log.csv'

# Read the data from the CSV file
step, train_loss, test_loss = read_csv_data(csv_filename)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(step, train_loss, label='Train Loss', color='blue')
plt.plot(step, test_loss, label='Test Loss', color='red')
plt.xlabel('Step Number')
plt.ylabel('Loss')
plt.title('Train and Test Loss Over Steps')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

