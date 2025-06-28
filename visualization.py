import plotly.graph_objects as go
import numpy as np
import open3d as o3d





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


def open3d_visualization(coord, feat=None):
    """
    Visualize a point cloud using Open3D.

    Parameters:
    - coord: (N, 3) numpy array of XYZ coordinates.
    - feats: (N, 3) numpy array of RGB colors in range [0, 1]. Optional.
    """
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError("coord must be of shape (N, 3)")

    if feat is not None:
        if feat.shape != coord.shape:
            raise ValueError("feats must be the same shape as coord")
        # Ensure RGB values are in [0, 1]
        if feat.max() > 1.0:
            feat = feat / 255.0
    else:
        # Use uniform gray if no color is provided
        feat = np.tile([0.5, 0.5, 0.5], (coord.shape[0], 1))

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(feat)
    # Visualize
    o3d.visualization.draw_geometries([pcd])




import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np

def visualize_two_point_clouds(coord_full, coord_down, feat_full=None, feat_down=None):
    # Normalize RGB
    if feat_full is not None and feat_full.max() > 1.0:
        feat_full = feat_full / 255.0
    if feat_down is not None and feat_down.max() > 1.0:
        feat_down = feat_down / 255.0

    # Create full-resolution point cloud
    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(coord_full)
    if feat_full is not None:
        pcd_full.colors = o3d.utility.Vector3dVector(feat_full)
    else:
        pcd_full.paint_uniform_color([0.5, 0.5, 0.5])

    # Create downsampled point cloud
    pcd_down = o3d.geometry.PointCloud()
    pcd_down.points = o3d.utility.Vector3dVector(coord_down)
    if feat_down is not None:
        pcd_down.colors = o3d.utility.Vector3dVector(feat_down)
    else:
        pcd_down.paint_uniform_color([1.0, 0.0, 0.0])  # Red for visibility

    # Open3D GUI app
    app = gui.Application.instance
    app.initialize()

    window = app.create_window("Synchronized Point Clouds", 1600, 800)
    scene_widget1 = gui.SceneWidget()
    scene_widget2 = gui.SceneWidget()

    # Setup first scene
    scene1 = rendering.Open3DScene(window.renderer)
    scene1.set_background([1, 1, 1, 1])
    scene1.add_geometry("full", pcd_full, rendering.MaterialRecord())
    scene_widget1.scene = scene1

    # Setup second scene
    scene2 = rendering.Open3DScene(window.renderer)
    scene2.set_background([1, 1, 1, 1])
    scene2.add_geometry("down", pcd_down, rendering.MaterialRecord())
    scene_widget2.scene = scene2

    # Set same camera pose
    bounds = pcd_full.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_extent().max()
    cam = scene1.camera
    cam.look_at(center, center + [0, 0, 1], [0, 1, 0])
    scene2.camera.copy_from(scene1.camera)

    # Layout the window with two views
    h_layout = gui.Horiz()
    h_layout.add_child(scene_widget1)
    h_layout.add_child(scene_widget2)
    window.add_child(h_layout)

    # Callback to sync camera
    def on_mouse_event(event):
        scene2.camera.copy_from(scene1.camera)
        return gui.Widget.EventCallbackResult.IGNORED

    scene_widget1.set_on_mouse(on_mouse_event)

    app.run()


def visualize_two_point_clouds2(coord_full, coord_down, feat_full=None, feat_down=None):
    # Normalize RGB if necessary
    if feat_full is not None and feat_full.max() > 1.0:
        feat_full = feat_full / 255.0
    if feat_down is not None and feat_down.max() > 1.0:
        feat_down = feat_down / 255.0

    # Create full-resolution point cloud
    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(coord_full)
    if feat_full is not None:
        pcd_full.colors = o3d.utility.Vector3dVector(feat_full)
    else:
        pcd_full.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color

    # Create downsampled point cloud
    pcd_down = o3d.geometry.PointCloud()
    pcd_down.points = o3d.utility.Vector3dVector(coord_down)
    if feat_down is not None:
        pcd_down.colors = o3d.utility.Vector3dVector(feat_down)
    else:
        pcd_down.paint_uniform_color([1.0, 0.0, 0.0])  # Red color

    # Initialize GUI app
    app = gui.Application.instance
    app.initialize()

    window = app.create_window("Synchronized Point Clouds", 1600, 800)
    scene_widget1 = gui.SceneWidget()
    scene_widget2 = gui.SceneWidget()

    # Create a lit material
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"  # Enables lighting

    # Setup first scene
    scene1 = rendering.Open3DScene(window.renderer)
    scene1.set_background([1, 1, 1, 1])
    #scene1.scene.set_lighting(scene1.scene.LightingProfile.NO_SHADOWS, [0, 0, 0])
    scene1.add_geometry("full", pcd_full, material)
    scene_widget1.scene = scene1

    # Setup second scene
    scene2 = rendering.Open3DScene(window.renderer)
    scene2.set_background([1, 1, 1, 1])
    #scene2.scene.set_lighting(scene2.scene.LightingProfile.NO_SHADOWS, [0, 0, 0])
    scene2.add_geometry("down", pcd_down, material)
    scene_widget2.scene = scene2

    # Setup camera using bounding box
    bounds = pcd_full.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    bounds = pcd_full.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_extent().max()

    scene_widget1.scene.camera.look_at(center, center + [0, 0, 1], [0, 1, 0])
    scene_widget2.scene.camera.copy_from(scene_widget1.scene.camera)


    #scene1.scene.camera.look_at(center, center + [0, 0, 1], [0, 1, 0])
    #scene2.scene.camera.copy_from(scene1.scene.camera)

    #scene1.setup_camera(60, bounds, center)
    #scene2.setup_camera(60, bounds, center)

    # Layout the window with two views side-by-side
    h_layout = gui.Horiz()
    h_layout.add_child(scene_widget1)
    h_layout.add_child(scene_widget2)
    window.add_child(h_layout)

    # Synchronize camera between views
    def on_mouse_event(event):
        scene2.scene.camera.copy_from(scene1.scene.camera)
        return gui.Widget.EventCallbackResult.IGNORED

    scene_widget1.set_on_mouse(on_mouse_event)

    app.run()


def visualize_two_point_clouds_overlay(coord_full, coord_down, feat_full=None, feat_down=None):
    if feat_full is not None and feat_full.max() > 1.0:
        feat_full = feat_full / 255.0
    if feat_down is not None and feat_down.max() > 1.0:
        feat_down = feat_down / 255.0

    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(coord_full)
    if feat_full is not None:
        pcd_full.colors = o3d.utility.Vector3dVector(feat_full)
    else:
        pcd_full.paint_uniform_color([0.5, 0.5, 0.5])  # gray

    pcd_down = o3d.geometry.PointCloud()
    pcd_down.points = o3d.utility.Vector3dVector(coord_down)
    if feat_down is not None:
        pcd_down.colors = o3d.utility.Vector3dVector(feat_down)
    else:
        pcd_down.paint_uniform_color([1.0, 0.0, 0.0])  # red

    # Visualize both in one window
    o3d.visualization.draw_geometries([pcd_full, pcd_down])