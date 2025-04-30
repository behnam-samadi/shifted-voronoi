import plotly.graph_objects as go
import numpy as np





def some_kind_of_visualization():


    feat = np.load('feat.npy')
    coord = np.load('coord.npy')


    #coord = coord[::10]
    #feat = feat[::10]

    # Convert coord to separate x, y, z
    x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]

    # Normalize RGB values if needed
    rgb = feat[:, :3]  # assuming RGB is in feat
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  # normalize to [0, 1]
    rgb_colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in rgb]

    # Optional: Use labels to color points instead of RGB
    # import matplotlib.pyplot as plt
    # cmap = plt.get_cmap('tab20')
    # rgb_colors = [f'rgb{tuple((np.array(cmap(l % 20)[:3]) * 255).astype(int))}' for l in label]

    # Create the scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                opacity=0.8
            )
        )
    ])

    fig.update_layout(
        title="Point Cloud Visualization",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()





import numpy as np
import plotly.graph_objects as go


def downsample_and_visualize(coord, idx):
    coord = coord.cpu()
    downsample_rate = 1

    # Step 1: Downsample coord
    coord_sub = coord[::downsample_rate]

    # Step 2: Get the original indices of coord_sub
    coord_sub_indices = np.arange(len(coord))[::downsample_rate]

    # Step 3: Find intersection (in original index space)
    intersected = np.intersect1d(coord_sub_indices, idx)

    # Step 4: Map those intersected indices back to coord_sub index space
    # We want: coord_sub[mapped_indices] == coord[intersected]
    mapped_indices = np.nonzero(np.isin(coord_sub_indices, intersected))[0]

    plot_style_2(coord_sub, mapped_indices)


    # plot_style_2(coord, idx_data[0])
def test_plot(cloud):
    # Assuming your point cloud is stored in the 'cloud' variable
    #cloud = np.random.rand(100, 3)  # Replace with your actual data

    # Define the indices of the points you want to color red
    red_indices = np.random.choice(range(cloud.shape[0]), 10, replace=False)

    # Create a figure
    fig = go.Figure(data=go.Scatter3d(
        x=cloud[:, 0],
        y=cloud[:, 1],
        z=cloud[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            line=dict(color='black', width=1),
            color=['blue' if i not in red_indices else 'red' for i in range(cloud.shape[0])]
        )
    ))

    # Show the figure
    fig.show()


def plot_style_2(cloud, special_indices=None):
    # Assuming your point cloud is stored in the 'cloud' variable
    # Define the indices of the points you want to color red
    if special_indices is None:
        special_indices = np.random.choice(range(cloud.shape[0]), 0, replace=False)



    # Create a figure
    fig = go.Figure(data=go.Scatter3d(
        x=cloud[:, 0],
        y=cloud[:, 1],
        z=cloud[:, 2],
        mode='markers',
        marker=dict(
            size=[4 if i in special_indices else 0 for i in range(cloud.shape[0])],
            line=dict(color='black', width=1),
            color=['red' if i in special_indices else 'blue' for i in range(cloud.shape[0])]
        )
    ))

    # Show the figure
    fig.show()




def plot_style_2_3colors(cloud, special_indices1=None, special_indices2 = None):
    # Assuming your point cloud is stored in the 'cloud' variable
    # Define the indices of the points you want to color red
    if special_indices1 is None:
        special_indices1 = np.random.choice(range(cloud.shape[0]), 0, replace=False)

    if special_indices2 is None:
        special_indices2 = np.random.choice(range(cloud.shape[0]), 0, replace=False)



    # Create a figure
    fig = go.Figure(data=go.Scatter3d(
        x=cloud[:, 0],
        y=cloud[:, 1],
        z=cloud[:, 2],
        mode='markers',
        marker=dict(
            size=[4 if i in special_indices1 else 4 for i in range(cloud.shape[0])],
            line=dict(color='black', width=1),
            color=['red' if i in special_indices1 else 'blue' if i in special_indices2 else 'green' for i in range(cloud.shape[0])]
        )
    ))

    # Show the figure
    fig.show()


def plot_pc(pc, second_pc=None, s=4, o=0.6):
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]], )
    fig.add_trace(
        go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode='markers', marker=dict(size=s, opacity=o)),
        row=1, col=1
    )
    if second_pc is not None:
        fig.add_trace(
            go.Scatter3d(x=second_pc[:, 0], y=second_pc[:, 1], z=second_pc[:, 2], mode='markers',
                         marker=dict(size=s, opacity=o)),
            row=1, col=2
        )
    fig.update_layout(scene_aspectmode='data')
    fig.show()
