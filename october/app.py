# Imports
import streamlit as st
from torchviz import make_dot
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


# Dataset Preparation
def true_function(x):
    return np.sin(x) * np.cos(x) * x


x = np.linspace(-2 * np.pi, 2 * np.pi, 400)
y_true = true_function(x)
y_noisy = y_true + np.random.normal(0, 0.1, y_true.shape)

x_train_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_train_tensor = torch.tensor(y_noisy, dtype=torch.float32).view(-1, 1)

# Split data into 80% training and 20% validation
split_idx = int(0.8 * len(x_train_tensor))
x_train_tensor, x_val_tensor = x_train_tensor[:split_idx], x_train_tensor[split_idx:]
y_train_tensor, y_val_tensor = y_train_tensor[:split_idx], y_train_tensor[split_idx:]

# Activation Function Mapping
ACTIVATION_FUNCTIONS = {
    "ReLU": nn.ReLU(),
    "Leaky ReLU": nn.LeakyReLU(),
    "Parameterised ReLU": nn.PReLU(),
    "Exponential ReLU": nn.ELU(),
    "Tanh": nn.Tanh(),
    "Sigmoid": nn.Sigmoid(),
    "Swish": lambda x: x * torch.sigmoid(x),  # Swish activation
}


# Neural Network definition
class ComplexNN(nn.Module):
    def __init__(
        self, l1_reg=0.0, l2_reg=0.0, dropout_rate=0.0, noise_std=0.0, activation="ReLU"
    ):
        super(ComplexNN, self).__init__()

        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.noise_std = noise_std
        self.activation_func = ACTIVATION_FUNCTIONS[activation]

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

    def regularisation_loss(self):
        l1_loss = sum(p.abs().sum() for p in self.parameters())
        l2_loss = sum(p.pow(2).sum() for p in self.parameters())
        return self.l1_reg * l1_loss + self.l2_reg * l2_loss


# Initialize weights
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -1, 1)
        m.bias.data.fill_(0.01)


# Training function with recording predictions and weights
# Modified training function to store weights
def train_model_and_store_predictions(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs=100,
    lr=0.01,
    use_early_stopping=True,
    patience=10,
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    predictions_list = []
    loss_values = []

    best_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    weights_list = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train) + model.regularisation_loss()
        loss.backward()
        optimizer.step()

        predictions_list.append(outputs.detach().numpy())
        loss_values.append(loss.item())

        # Early stopping logic
        if use_early_stopping:
            with torch.no_grad():
                val_outputs = model(x_val)
                val_loss = criterion(val_outputs, y_val)
                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_without_improvement = 0
                    best_model_state = model.state_dict().copy()
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement == patience:
                        model.load_state_dict(best_model_state)
                        break

        weights_list.append([p.data.clone() for p in model.parameters()])

    return predictions_list, loss_values, weights_list


# Plotly Animations
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

    frames = []

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

    fig.frames = frames
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

    # fig.show()
    return fig


# Plotting the Loss
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

    # fig.show()
    return fig


# Visualize Neural Network
def visualize_nn(model, x):
    y = model(x)
    return make_dot(y, params=dict(list(model.named_parameters())))


# Function to plot weight distribution
def plot_weight_distribution(weights, epoch):
    # Visualize the weights of the first layer for simplicity
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=weights[0].view(-1).numpy(), name=f"Epoch {epoch}"))
    fig.update_layout(title=f"Weights Distribution at Epoch {epoch}")
    return fig


# New function to animate weight distribution over epochs
def weight_distribution_animation(weights_list):
    """
    Generate an animation of weight distributions over epochs.

    Args:
    - weights_list (list of list of torch.Tensor): A list where each element
      is another list containing the weights for each layer at that epoch.

    Returns:
    - fig (plotly.graph_objects.Figure): Animated figure of weight distributions.
    """
    # Initializing the figure
    fig = go.Figure()

    # Create histograms for each epoch
    frames = []
    for epoch, weights_in_epoch in enumerate(weights_list):
        # Flatten weights for the current epoch
        flattened_weights = []
        for weight_tensor in weights_in_epoch:
            flattened_weights.extend(weight_tensor.cpu().numpy().flatten())

        frame = go.Frame(
            data=[
                go.Histogram(
                    x=flattened_weights,
                    name="Weight Distribution",
                    nbinsx=50,  # Adjust this based on your preference
                )
            ],
            name=str(epoch),
        )
        frames.append(frame)

    # Add frames to the figure
    fig.frames = frames

    # Define the layout
    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
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
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=[
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
                "steps": [
                    {
                        "args": [
                            [f.name],
                            {
                                "frame": {"duration": 300, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 300},
                            },
                        ],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(fig.frames)
                ],
            }
        ],
    )

    fig.update_layout(
        title="Weight Distribution Over Epochs",
        xaxis_title="Weight Value",
        yaxis_title="Frequency",
        xaxis=dict(gridcolor="#808080"),
        yaxis=dict(gridcolor="#808080"),
        plot_bgcolor="#171717",
        paper_bgcolor="#171717",
        font=dict(color="white"),
        bargap=0.05,  # Gap between bars
    )

    return fig


# Main function to be run when the Streamlit app is executed
def main():
    st.title("Neural Network Visualisations")
    st.write('Salmaan Suliman')

    # Sidebar for user inputs
    st.sidebar.header("Settings")

    # Custom True Function Input
    custom_function = st.sidebar.text_input(
        "Custom True Function (using 'np' for numpy):", "np.sin(x) * np.cos(x) * x"
    )

    try:
        global true_function
        true_function = lambda x: eval(custom_function)
        st.sidebar.text("Function updated successfully!")
    except Exception as e:
        st.sidebar.text(f"Error in function: {e}")

    # Re-calculate y based on the new function
    global y_true, y_noisy, x_train_tensor, y_train_tensor
    y_true = true_function(x)
    y_noisy = y_true + np.random.normal(0, 0.1, y_true.shape)
    x_train_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y_train_tensor = torch.tensor(y_noisy, dtype=torch.float32).view(-1, 1)

    num_epochs = st.sidebar.slider("Number of Epochs", 10, 200, 100)

    # Model Parameters
    l1_reg = st.sidebar.slider("L1 Regularisation", 0.0, 0.1, 0.01, 0.01)
    l2_reg = st.sidebar.slider("L2 Regularisation", 0.0, 0.1, 0.01, 0.01)
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 1.0, 0.5, 0.1)
    noise_std = st.sidebar.slider("Noise Standard Deviation", 0.0, 1.0, 0.1, 0.1)

    activation = st.sidebar.selectbox(
        "Activation Function", list(ACTIVATION_FUNCTIONS.keys())
    )

    early_stopping = st.sidebar.checkbox("Use Early Stopping", value=True)
    patience = st.sidebar.slider("Early Stopping Patience", 1, 50, 10)

    # Select a single model
    selected_model = st.sidebar.selectbox(
        "Choose a model",
        options=[
            "No Regularisation",
            "L1 Regularisation",
            "L2 Regularisation",
            "L1 & L2 Regularisation",
            "Dropout",
            "Noise Injection",
            "Custom Model",
        ],
    )

    # Second model selectbox
    selected_model2 = st.sidebar.selectbox(
        "Choose a second model to compare",
        options=[
            "None",
            "No Regularisation",
            "L1 Regularisation",
            "L2 Regularisation",
            "L1 & L2 Regularisation",
            "Dropout",
            "Noise Injection",
            "Custom Model",
        ],
    )

    # Mapping of model names to their instances
    model_mapping = {
        # Mapping of model names to their instances
        "No Regularisation": ComplexNN(),
        "L1 Regularisation": ComplexNN(l1_reg=0.01),
        "L2 Regularisation": ComplexNN(l2_reg=0.01),
        "L1 & L2 Regularisation": ComplexNN(l1_reg=0.01, l2_reg=0.01),
        "Dropout": ComplexNN(dropout_rate=0.5),
        "Noise Injection": ComplexNN(noise_std=0.1),
        "Custom Model": ComplexNN(
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate,
            noise_std=noise_std,
            activation=activation,
        ),
    }

    model_instance = model_mapping[selected_model]

    # Training and visualization for the first model
    st.sidebar.text(f"Training {selected_model}...")
    predictions_list, loss_values, weights_list = train_model_and_store_predictions(
        model_instance,
        x_train_tensor,
        y_train_tensor,
        x_val_tensor,
        y_val_tensor,
        epochs=num_epochs,
        use_early_stopping=early_stopping,
        patience=patience,
    )

    st.plotly_chart(plotly_animation(x, y_true, y_noisy, predictions_list))

    # If second model is selected, train and visualize it
    if selected_model2 != "None":
        model_instance2 = model_mapping[selected_model2]

        st.sidebar.text(f"Training {selected_model2}...")
        (
            predictions_list2,
            loss_values2,
            weights_list2,
        ) = train_model_and_store_predictions(
            model_instance2,
            x_train_tensor,
            y_train_tensor,
            x_val_tensor,
            y_val_tensor,
            epochs=num_epochs,
            use_early_stopping=early_stopping,
            patience=patience,
        )

        st.subheader(f"Comparison between {selected_model} and {selected_model2}")

        st.plotly_chart(plotly_animation(x, y_true, y_noisy, predictions_list2))

    st.subheader(f"{selected_model} - Loss over Epochs")
    st.plotly_chart(plot_loss_graph(loss_values, selected_model))
    # Visualize Neural Network
    st.subheader(f"{selected_model} - Neural Network Visualisation")
    st.graphviz_chart(str(visualize_nn(model_instance, x_train_tensor)))

    # Animated weight distribution
    st.subheader(f"{selected_model} - Weight Distribution over Epochs")
    st.plotly_chart(weight_distribution_animation(weights_list))


# Run the Streamlit app
if __name__ == "__main__":
    main()
