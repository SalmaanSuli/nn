# Imports
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import plotly.graph_objects as go


# --------------------------------------------------------------------------------------------------------------------------------------
# Dataset Preparation
def true_function(x):
    return np.sin(x) * np.cos(x) * x


x = np.linspace(-2 * np.pi, 2 * np.pi, 400)
y_true = true_function(x)
y_noisy = y_true + np.random.normal(0, 0.1, y_true.shape)

x_train_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_train_tensor = torch.tensor(y_noisy, dtype=torch.float32).view(-1, 1)


# --------------------------------------------------------------------------------------------------------------------------------------
# Neural Network definition
class ComplexNN(nn.Module):
    def __init__(
        self, l1_reg=0.0, l2_reg=0.0, dropout_rate=0.0, noise_std=0.0, activation="relu"
    ):
        super(ComplexNN, self).__init__()

        # Regularization parameters
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.noise_std = noise_std
        self.activation_func = nn.ReLU() if activation == "relu" else nn.Tanh()

        # Neural Network architecture
        self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(32, 1)

        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.batch_norm4 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if self.noise_std > 0:
            x += torch.normal(0, self.noise_std, size=x.shape)

        x = self.activation_func(self.batch_norm1(self.layer1(x)))
        x = self.activation_func(self.batch_norm2(self.layer2(x)))
        x = self.dropout(x)
        x = self.activation_func(self.batch_norm3(self.layer3(x)))
        x = self.activation_func(self.batch_norm4(self.layer4(x)))
        x = self.layer5(x)
        return x

    # Regularization Loss
    def regularisation_loss(self):
        l1_loss = sum(p.abs().sum() for p in self.parameters())
        l2_loss = sum(p.pow(2).sum() for p in self.parameters())
        return self.l1_reg * l1_loss + self.l2_reg * l2_loss


# --------------------------------------------------------------------------------------------------------------------------------------
# Initialize weights
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -1, 1)
        m.bias.data.fill_(0.01)


# --------------------------------------------------------------------------------------------------------------------------------------
# Training function with recording predictions
def train_model_and_store_predictions(model, x_train, y_train, epochs=100, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    predictions_list = []
    loss_values = []  # Store loss values at each epoch

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = (
            criterion(outputs, y_train) + model.regularisation_loss()
        )  # Add regularization loss
        loss.backward()
        optimizer.step()
        predictions_list.append(outputs.detach().numpy())
        loss_values.append(loss.item())  # Append the loss value

    return predictions_list, loss_values  # Return both predictions and loss values


# --------------------------------------------------------------------------------------------------------------------------------------
# For Streamlit, instead of showing the plots directly, return the figures


def plotly_animation(x, y_true, y_noisy, predictions):
    # Create figure
    fig = go.Figure()

    # True function
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_true,
            mode="lines",
            name="True Function",
            line=dict(color="#037493", width=2),
        )
    )

    # Noisy data
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_noisy,
            mode="markers",
            name="Noisy Data",
            marker=dict(color="#FF0000", size=2),
        )
    )

    # Sliders and buttons for animation control
    sliders = [
        {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Epoch:",
                "visible": True,
                "xanchor": "right",
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [],
        }
    ]

    frames = []  # List to hold our frames

    for i, y_pred in enumerate(predictions):
        frame = {
            "data": [
                go.Scatter(
                    x=x,
                    y=y_pred.squeeze(),
                    mode="lines",
                    name="NN Prediction",
                    line=dict(color="#ADD8E6", width=3),
                )
            ],
            "name": str(i),
        }
        frames.append(frame)
        slider_step = {
            "args": [
                [str(i)],
                {
                    "frame": {"duration": 300, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 300},
                },
            ],
            "label": str(i),
            "method": "animate",
        }
        sliders[0]["steps"].append(slider_step)

    fig.frames = frames  # Assign our list of frames to the figure
    fig.update_layout(sliders=sliders)

    # Adding animation control buttons
    play_button = {
        "args": [
            None,
            {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True},
        ],
        "label": "Play",
        "method": "animate",
    }

    pause_button = {
        "args": [
            [None],
            {
                "frame": {"duration": 0, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": 0},
            },
        ],
        "label": "Pause",
        "method": "animate",
    }

    fig.update_layout(
        updatemenus=[
            {
                "buttons": [play_button, pause_button],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ]
    )
    # Layout
    fig.update_layout(
        title=f"Neural Network Function Approximation",
        xaxis_title="X",
        yaxis_title="Y",
        xaxis=dict(gridcolor="#808080"),
        yaxis=dict(gridcolor="#808080"),
        plot_bgcolor="#171717",
        paper_bgcolor="#171717",
        font=dict(color="white"),
    )

    fig.show()
    return fig


# --------------------------------------------------------------------------------------------------------------------------------------


def plot_loss_graph(loss_values, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_values, mode="lines", name="Loss Value"))

    fig.update_layout(
        title=f"{title} - Loss over Epochs",
        xaxis_title="Epochs",
        yaxis_title="Loss Value",
        xaxis=dict(gridcolor="#808080"),
        yaxis=dict(gridcolor="#808080"),
        plot_bgcolor="#171717",
        paper_bgcolor="#171717",
        font=dict(color="white"),
    )
    fig.show()

    return fig


# --------------------------------------------------------------------------------------------------------------------------------------


# Main function to be run when the Streamlit app is executed
def main():
    st.title("Neural Network Visualisations (S. Suliman)")

    # Sidebar for user inputs
    st.sidebar.header("Settings")
    num_epochs = st.sidebar.slider("Number of Epochs", 10, 200, 100)
    selected_model = st.sidebar.selectbox(
        "Choose a model",
        options=[
            "No Regularisation",
            "L1 Regularisation",
            "L2 Regularisation",
            "L1 & L2 Regularisation",
            "Dropout",
            "Noise Injection",
        ],
    )

    # Create a dictionary to map model names to their instances
    model_mapping = {
        "No Regularisation": ComplexNN(),
        "L1 Regularisation": ComplexNN(l1_reg=0.01),
        "L2 Regularisation": ComplexNN(l2_reg=0.01),
        "L1 & L2 Regularisation": ComplexNN(l1_reg=0.01, l2_reg=0.01),
        "Dropout": ComplexNN(dropout_rate=0.5),
        "Noise Injection": ComplexNN(noise_std=0.1),
    }

    st.sidebar.text(f"Training {selected_model}...")
    model_instance = model_mapping[selected_model]
    predictions_list, loss_values = train_model_and_store_predictions(
        model_instance, x_train_tensor, y_train_tensor, epochs=num_epochs
    )

    # Display plots
    st.subheader("Testing and Visualising NN for Function Approximation")
    st.plotly_chart(plotly_animation(x, y_true, y_noisy, predictions_list))

    st.subheader(f"{selected_model} - Loss over Epochs")
    st.plotly_chart(plot_loss_graph(loss_values, selected_model))


if __name__ == "__main__":
    main()
