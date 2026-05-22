# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="642" height="829" alt="image" src="https://github.com/user-attachments/assets/4f30fedd-ad42-46b9-bcbd-f7cac020cfdc" />


## DESIGN STEPS
### Step 1: Load and Understand the Dataset

Load the customer segmentation dataset and study the features such as age, gender, profession, spending score, and other customer details to understand the data structure.

### Step 2: Data Preprocessing

Remove unnecessary columns like ID, handle missing values, and convert categorical data into numerical form using Label Encoding or One-Hot Encoding.

### Step 3: Split and Scale the Data

Separate the dataset into training and testing sets. Apply feature scaling using StandardScaler to normalize the input values for better model performance.

### Step 4: Build the Neural Network Model

Design a feedforward neural network with input, hidden, and output layers using activation functions like ReLU and Softmax for multi-class customer segmentation.

### Step 5: Train the Model

Train the neural network using the training dataset with suitable loss functions and optimizers. Perform forward propagation, backpropagation, and weight updates for multiple epochs.

### Step 6: Evaluate and Predict Customer Segments

Test the trained model using test data, evaluate accuracy and classification metrics, and predict the customer segment (A, B, C, or D) for new customers.

## PROGRAM

### Name:Nisha J

### Register Number:212223040133

```
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        #Include your code here
        self.fc1=nn.Linear(input_size, 32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,4)
    def forward(self,x):
      x=F.relu(self.fc1(x))
      x=F.relu(self.fc2(x))
      x=F.relu(self.fc3(x))
      x=self.fc4(x)
      return x
        
# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
  #Include your code here
  model.train()
  for epoch in range(epochs):
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs=model(inputs)
      loss=criterion(outputs, labels)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Initialize model
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

train_model(model,train_loader,criterion,optimizer,epochs=100)


```

### Dataset Information
<img width="1405" height="273" alt="image" src="https://github.com/user-attachments/assets/9ff14be6-1b2c-4c09-9d0a-713a0405ce57" />


### OUTPUT

## Confusion Matrix
<img width="759" height="600" alt="image" src="https://github.com/user-attachments/assets/34b31714-0b20-4b72-8d1d-03e2a0004b7f" />


## Classification Report
<img width="673" height="497" alt="image" src="https://github.com/user-attachments/assets/9880d65f-c5b3-4d48-837a-e2667f8afdd4" />


### New Sample Data Prediction
<img width="419" height="111" alt="image" src="https://github.com/user-attachments/assets/a3862f10-5839-46e0-8420-c1712aaca4b5" />


## RESULT
Thus developing a neural network classification model for the given data set has been executed successfully.
