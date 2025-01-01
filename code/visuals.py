from fish_utils import *

# ================================ plotting functions ======================================

# plain frame plots
def plot_fish_frame(frame_of_interest, center_method='least_squares', filepath='erikas_1min.csv', figsize=(8,8)):
    """ matplotlib polar plot of fish skeletons at a specific frame of the video """

    df = load_and_preprocess_data(filepath, center_method)

    # filtering for frame of interest
    df_frame = df[df['frame_idx'] == frame_of_interest]
    max_radius = df[['mouth_radius', 'L_eye_radius', 'R_eye_radius', 'tail_radius', 'spine_radius']].max().max()

    # prep to plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
    parts = ['mouth', 'L_eye', 'R_eye', 'tail', 'spine']
    edges = [('L_eye', 'mouth'), ('R_eye', 'mouth'), ('L_eye', 'spine'), ('R_eye', 'spine'), ('spine', 'tail')]

    # plot each part and draw edges
    for track_id, group in df_frame.groupby('track'):
        color = standard_colors[int(track_id[-1])]  # Assuming track_id is like 'track_0'
        for part in parts:
            ax.scatter(group[f'{part}_theta'], group[f'{part}_radius'], label=part if track_id == 'track_0' else "", alpha=0.75, color=color)
        for edge in edges:
            part1, part2 = edge
            ax.plot([group.iloc[0][f'{part1}_theta'], group.iloc[0][f'{part2}_theta']],
                    [group.iloc[0][f'{part1}_radius'], group.iloc[0][f'{part2}_radius']], color=color)

    # final adjustments
    ax.set_ylim(0, max_radius)  # Set radial limits
    ax.set_title(f'Polar Plot of Fish Skeletons at Frame {frame_of_interest}')
    ax.set_xlabel('Theta (radians)')
    ax.set_ylabel('Radius (pixels)', labelpad=30, rotation=-270, va='bottom')
    ax.grid(True)
    plt.tight_layout()
    plt.show()




def plotly_fish_frame(frame_of_interest, center_method='least_squares', filepath='erikas_1min.csv', width=850, height=850):
    """ plotly polar plot of fish skeletons at a specific frame of the video """

    df = load_and_preprocess_data(filepath, center_method)

    # filter data for the specific frame
    df_frame = df[df['frame_idx'] == frame_of_interest]
    max_radius = df[['mouth_radius', 'L_eye_radius', 'R_eye_radius', 'tail_radius', 'spine_radius']].max().max()

    # create the Plotly figure with polar layout
    fig = go.Figure()

    # define body parts and edges for plotting
    parts = ['mouth', 'L_eye', 'R_eye', 'tail', 'spine']
    edges = [('L_eye', 'mouth'), ('R_eye', 'mouth'), ('L_eye', 'spine'), ('R_eye', 'spine'), ('spine', 'tail')]

    # Track used legend to avoid duplicate legends for the same track
    track_used_legend = []

    # plot each part and draw edges
    for track_id, group in df_frame.groupby('track'):
        color = px_standard_colors[int(track_id[-1])]  # Assuming track_id is like 'track_0'
        for part in parts:
            # adjust theta for Plotly's requirements (found this through trial + error...if it works, it works)
            theta_adjusted = -(np.rad2deg(group[f'{part}_theta']) + 270) % 360
            fig.add_trace(go.Scatterpolar(
                r=group[f'{part}_radius'],
                theta=theta_adjusted,
                mode='markers',
                name=f'{track_id}' if track_id not in track_used_legend else '',
                marker=dict(color=color, size=9, opacity = 0.75),
                legendgroup=track_id,  # Grouping legends by track id
                showlegend=track_id not in track_used_legend
            ))
            if track_id not in track_used_legend:
                track_used_legend.append(track_id)
        for edge in edges:
            part1, part2 = edge
            theta1_adjusted = -(np.rad2deg(group.iloc[0][f'{part1}_theta']) + 270) % 360
            theta2_adjusted = -(np.rad2deg(group.iloc[0][f'{part2}_theta']) + 270) % 360
            fig.add_trace(go.Scatterpolar(
                r=[group.iloc[0][f'{part1}_radius'], group.iloc[0][f'{part2}_radius']],
                theta=[theta1_adjusted, theta2_adjusted],
                mode='lines',
                line=dict(color=color),
                legendgroup=track_id,
                showlegend=False
            ))

    # update the layout for polar plot specifics, placing axes below the traces
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max_radius], layer='below traces'),  # add buffer to radial axis range
            angularaxis=dict(direction='clockwise', thetaunit='degrees', layer='below traces')
        ),
        title=f'Polar Plot of Fish Skeletons at Frame {frame_of_interest}',
        width=850,
        height=850
    )
    # show the figure
    fig.show()






# plain distance plots
def plot_distance(ax, distance_df, track_pair_name, frame_range=(0, 1000), rolling_avg=False, color='teal', alpha=0.5, linewidth=1.5, linestyle='dashed'):
    """
    plot distance between specified tracks over desired frame range

    inputs:
        ax: Axes object on which to plot (allows for overlaying multiple plots).
        distance_df: DataFrame with distance data (preprocessed)
        track_pair_name (str): Label for the plot legend.
        frame_range (tuple): Tuple of (start_frame, end_frame) to define the range of frames to plot.
        rolling_avg (bool): If True, plot the rolling average distance instead of raw distance.
        color (str): Color of the plot line.
        alpha (float): Transparency of the plot line.
        linestyle (str): Style of the plot line.

    returns:
        None. Helper function to plot on the provided axes.


    example use:
        distance_df =  calc_dist_for_pair(df, 'track_0', 'track_1', window_size=50)
        fig, ax = plt.subplots(figsize=(14, 4), layout='constrained')
        plot_distance(ax, distance_df, 'Track 0 to Track 1', frame_range=(500, 1000), rolling_avg=False, color='purple')
        plot_distance(ax, distance_df, 'Track 0 to Track 1', frame_range=(500, 1000), rolling_avg=True, color='violet')
        plt.show()
    """

    # ensure we sort by frame index
    data_sorted = distance_df.sort_values('frame_idx', ascending=True)

    # filter for range
    data_filtered = data_sorted[(data_sorted['frame_idx'] >= frame_range[0]) & (data_sorted['frame_idx'] <= frame_range[1])]

    # to help us plot
    xs = data_filtered['frame_idx']
    ys = data_filtered['rolling_avg_distance'] if rolling_avg else data_filtered['distance']

    # plotting
    ax.plot(xs, ys, label=f"{track_pair_name} {'Rolling Avg' if rolling_avg else 'Raw'}", color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Distance')
    ax.legend()

    return data_filtered


def plot_distance_distribution(distance_df, title, color='blue', xlim=None):
    """
    plot the distribution of distances with a histogram and KDE.

    inputs:
    distance_df (DataFrame): df containing 'distance' column.
    title (str): title of the plot.
    color (str): color of the histogram.
    xlim (tuple): X-axis limits for the plot.
    """
    plt.figure(figsize=(7, 4))
    sns.histplot(distance_df['distance'], kde=True, bins='auto', color=color)
    plt.xlabel('Distance')
    plt.ylabel('Density')
    if xlim:
        plt.xlim(xlim)
    plt.title(title)
    plt.show()



def overlay_distance_distributions(dist_dfs, labels, colors, alpha=0.6):
    """
    overlay histograms of distance distributions.

    inputs:
    dist_dfs (list): List of DataFrames, each containing 'distance' data.
    labels (list): Labels for each distribution in the legend.
    colors (list): Colors for each histogram.
    alpha (float): Opacity of the histograms.
    """
    plt.figure(figsize=(12, 4))
    for df, label, color in zip(dist_dfs, labels, colors):
        sns.histplot(df['distance'], color=color, bins=100, kde=False, label=label, alpha=alpha)

    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.title('Overlay of Distance Distributions')
    plt.legend()
    plt.show()

def plot_avg_center_dists(frame_of_interest, center_method='least_squares', filepath = 'erikas_1min.csv', figsize=(12, 10)):
    """
    plots the average distances of each fish to the center (over all frames of video) with a specific frame as the background

    parameters:
    file_path - Path to the CSV file containing the data
    frame_of_interest - The frame index to plot
    center_method - The method used for centering (default 'least_squares')
    figsize - Size of the figure (default (12, 10))
    """
    plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "black",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "white",
    "xtick.color": "black",
    "ytick.color": "#747474",
    "grid.color": "#DFDFDF",
    "figure.facecolor": "white",
    "figure.edgecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white"
    })

    # load and preprocess data
    df = load_and_preprocess_data(filepath, center_method)
    max_radius = df[['mouth_radius', 'L_eye_radius', 'R_eye_radius', 'tail_radius', 'spine_radius']].max().max()
    df_frame = df[df['frame_idx'] == frame_of_interest]

    # calc average distances to center
    avg_distances_dict = calc_avg_center_dist(df)

    # create polar plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})

    # plot fish parts, connections, and average distances
    for track_id, group in df_frame.groupby('track'):
        color = standard_colors[int(track_id[-1])]
        for part in ['mouth', 'L_eye', 'R_eye', 'tail', 'spine']:
            ax.scatter(group[f'{part}_theta'], group[f'{part}_radius'], alpha=0.75, color=color)
        for edge in [('L_eye', 'mouth'), ('R_eye', 'mouth'), ('L_eye', 'spine'), ('R_eye', 'spine'), ('spine', 'tail')]:
            part1, part2 = edge
            ax.plot([group[f'{part1}_theta'], group[f'{part2}_theta']],
                    [group[f'{part1}_radius'], group[f'{part2}_radius']], color=color)

        # draw average distance circle for this track
        avg_radius = avg_distances_dict[track_id]
        circle = plt.Circle((0, 0), avg_radius, transform=ax.transData._b, color=color, fill=False, linestyle='--', linewidth=3, alpha=0.4)
        ax.add_artist(circle)

    # plot bars for average distances
    angles = np.linspace(0, -2.8/3 * np.pi, len(avg_distances_dict) + 2) # just asesthetics
    for i, (track, distance) in enumerate(avg_distances_dict.items()):
        start_angle = angles[i]
        end_angle = angles[i + 1]
        ax.bar((start_angle + end_angle) / 2, distance, width=(end_angle - start_angle) / 3, color=standard_colors[int(track[-1])], alpha=0.1)

    # set plot parameters
    ax.set_ylim(0, max_radius)
    ax.grid(True)
    ax.set_yticklabels('') # this gets rid of the pixel distance labels 100,200,300,400,500

    # add legend
    legend_entries = [plt.Line2D([0], [0], color=color, label=f'Fish {track_id[-1]}: {avg_distances_dict[track_id]:.2f}', linestyle='-', alpha=0.4, linewidth=6) for track_id, color in zip(avg_distances_dict.keys(), standard_colors)]
    ax.legend(handles=legend_entries, title='Average Distances', loc='lower left', fontsize=11, framealpha=0.9)
    fig.suptitle("Average Distance of Each Fish to the Center")
    plt.title("in a Lowe's 'United Solutions 5-Gal Food-grade Plastic General Bucket' (a nice bucket, indeed)", fontsize=10, style='italic', pad=20)  # subtitle, with padding for position

    plt.show()


def plotly_fish_frame_with_trails(frame_of_interest, center_method='least_squares', filepath='erikas_1min.csv', title=general, trail_length=120, height=850, width=700):
    """ polar plot of fish skeletons at a specific frame with a trail from prior frames using plotly

        inputs:
        frame_of_interest - The frame index to plot
        center_method - The method used for centering (default 'least_squares')
        filepath - Path to the CSV file containing the data
    """
    df = load_and_preprocess_data(filepath)

    fig = go.Figure()
    parts = ['mouth', 'L_eye', 'R_eye', 'tail', 'spine']
    edges = [('L_eye', 'mouth'), ('R_eye', 'mouth'), ('L_eye', 'spine'), ('R_eye', 'spine'), ('spine', 'tail')]
    for i, track_id in enumerate(track_ids):

        # create a trace for each track
        # filter data for the specific track and for the frame range of interest
        df_track = df[(df['track'] == track_id) & (df['frame_idx'] <= frame_of_interest) & (df['frame_idx'] > frame_of_interest - trail_length)]
        df_track = df_track.sort_values('frame_idx')  # Ensure the DataFrame is sorted by frame index

        if df_track.empty:
            continue  # skip this track if there's no data in the range

        # correctly adjust the theta for Plotly (convert radians to degrees and adjust for polar orientation)
        adjusted_theta = -(np.rad2deg(df_track['spine_theta']) + 270) % 360  # Normalize to ensure the angle stays within 0-360

        # add the trace for the trail of each track to the figure
        fig.add_trace(go.Scatterpolar(
            r=df_track['spine_radius'],
            theta=adjusted_theta,
            mode='lines',
            name=track_id,
            line=dict(color=set_opacity(px_standard_colors[i]), dash='solid', width=7),  # Adjusted line width and style
        ))

        # add the skeleton for the specific frame
        df_frame = df[(df['track'] == track_id) & (df['frame_idx'] == frame_of_interest)]

        df_frame = df[(df['track'] == track_id) & (df['frame_idx'] == frame_of_interest)]
        if df_frame.empty:
            print(f"No data for track {track_id} at frame {frame_of_interest}")
            continue  # Skip if no data for this frame

        for part in parts:
            fig.add_trace(go.Scatterpolar(
                r=[df_frame[f'{part}_radius'].values[0]],
                theta=[-(np.rad2deg(df_frame[f'{part}_theta'].values[0]) + 270) % 360],
                mode='markers',
                marker=dict(color=px_standard_colors[i], size=8),
                showlegend=False
            ))
        for edge in edges:
            part1, part2 = edge
            fig.add_trace(go.Scatterpolar(
                r=[df_frame[f'{part1}_radius'].values[0], df_frame[f'{part2}_radius'].values[0]],
                theta=[-(np.rad2deg(df_frame[f'{part1}_theta'].values[0]) + 270) % 360, -(np.rad2deg(df_frame[f'{part2}_theta'].values[0]) + 270) % 360],
                mode='lines',
                line=dict(color=px_standard_colors[i]),
                showlegend=False
            ))

    # adjustments
    fig.update_layout(

        polar=dict(
            radialaxis=dict(visible=True, range=[0, df['spine_radius'].max()], layer='below traces'),
            angularaxis=dict(direction='clockwise', thetaunit='degrees')
        ),
        title={
            'text': f"Polar Plot of Fish Skeletons at Frame {frame_of_interest}<br><sub><i>{title}</i></sub>",

            'x': 0.5,  # center the title
            'xanchor': 'center',  # anchor title at center horizontally
            'yanchor': 'top',  # anchor title at top vertically
            'y': 0.95  # adjust this value to add padding (lower values bring the title lower)
        },
        width=width,
        height=height
    )

    # show the figure
    fig.show()
    return fig

# spatial preference plot
def plot_kde_circle(filepath='erikas_1min.csv', center_method='least_squares', figsize=(10, 10)):
    """
    plot KDEs of fish spatial preferences with a circular boundary overlay (after conversion to cartesian coords)

    note: plot_kde_square was the precursor to this function (but did not display a circular boundary)
            kde does not naturally support polar coordinates, so this was my 'workaround'


    inputs:
    filepath - path to the CSV file containing the data
    center_method - the method used for centering (default 'least_squares')
    figsize - size of the figure (default (10, 10))
    """

    df = load_and_preprocess_data(filepath, center_method) # load and preprocess data
    df['x'], df['y'] = zip(*df.apply(lambda row: polar_to_cartesian(row['spine_radius'], row['spine_theta']), axis=1))

    sns.set_style("white")
    max_radius = df[['mouth_radius', 'L_eye_radius', 'R_eye_radius', 'tail_radius', 'spine_radius']].max().max()

    # create figure and axes for a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs = axs.flatten()
    cmaps = ['Blues', 'OrRd', 'YlOrBr', 'Purples']
    tracks = ['track_0', 'track_1', 'track_2', 'track_3']


    for i, track in enumerate(tracks):
        track_data = df[df['track'] == track]
        ax = axs[i]

        sns.kdeplot(
            x=track_data['x'], y=track_data['y'],
            ax=ax, cmap=cmaps[i], levels=15, thresh=0.3,
            fill=False, linewidths=2)

        # create a circular boundary
        circle = patches.Circle((0, 0), max_radius, edgecolor='#6F6F6F', facecolor='none', linestyle='-', linewidth=1.2)
        ax.add_patch(circle)

        # set the limits based on max_radius, allowing more space around the plots
        ax.set_xlim([-1.2 * max_radius, 1.2 * max_radius])
        ax.set_ylim([-1.08 * max_radius, 1.08 * max_radius])
        ax.set_aspect('equal')

        ax.set_title(f"Density Plot for Fish {track[-1]}", fontsize=10)
        ax.set_xlabel("X Coordinate", fontsize=10)
        ax.set_ylabel("Y Coordinate", fontsize=10)
        ax.grid(False)


    for ax in axs:
        ax.axis('off')

    # adjustments
    plt.subplots_adjust(hspace=0.05, wspace=0.1)
    plt.suptitle('KDE Plots of Fish Spatial Preferences', fontsize=13, verticalalignment='top')
    fig.text(0.52, 0.95, "in a Lowe's 'United Solutions 5-Gal Food-grade Plastic General Bucket' (a nice bucket, indeed)", ha='center', va='top', fontsize=10, style='italic')

    plt.show()

# plot_kde_circle(filepath='erikas_1min.csv', center_method='least_squares', figsize=(10, 10))        # spatial preferences

# trajectory (path) plots

def plot_4_paths(filepath='erikas_1min.csv', frame_start=500, frame_end=750):
    """
    use matplotlib to plot the paths of multiple fish in polar coordinates
    note: plotly_4_paths does the same thing but with plotly (and is prettier)


    inputs:
    filepath - path to the CSV file containing the data
    frame_start (int): starting frame index for plotting. default is 500.
    frame_end (int): emdomg frame index for plotting. default is 750.
    """
    df = load_and_preprocess_data(filepath=filepath, center_method='least_squares')
    max_radius = df[['mouth_radius', 'L_eye_radius', 'R_eye_radius', 'tail_radius', 'spine_radius']].max().max()

    # Create a polar plot
    fig, ax = plt.subplots(figsize=(6, 8), subplot_kw={'projection': 'polar'})

    # Define the tracks
    tracks = df['track'].unique()

    # Plotting loop for each track
    for i, track_id in enumerate(tracks):
        # Filter the DataFrame for the specific track and frame range
        df_track = df[(df['track'] == track_id) & (df['frame_idx'] >= frame_start) & (df['frame_idx'] <= frame_end)]
        df_track = df_track.sort_values('frame_idx')  # Ensure sorted by frame_idx if sequence matters

        # Plotting the path of the fish
        ax.plot(df_track['spine_theta'], df_track['spine_radius'], label=f'{track_id}', color=standard_colors[i], linestyle='-', linewidth=1)

        # Adding markers for the start (*) and end (<) of the track
        if not df_track.empty:
            # Start marker
            ax.plot(df_track.iloc[0]['spine_theta'], df_track.iloc[0]['spine_radius'], '.', markersize=8, color=standard_colors[i])
            # End marker
            ax.plot(df_track.iloc[-1]['spine_theta'], df_track.iloc[-1]['spine_radius'], '>', markersize=8, color=standard_colors[i])

    # Customizing the plot
    ax.set_ylim(0, max_radius if max_radius else df['spine_radius'].max())  # Set radial limits to the maximum radius found in your data
    ax.set_title(f'Paths for Multiple Tracks from Frame {frame_start} to {frame_end}', fontsize=12)
    ax.set_xlabel('Theta (radians)')
    ax.set_ylabel('Radius (pixels)', labelpad=30, rotation=-270, va='bottom')
    ax.set_yticklabels([])
    ax.grid(True)

    # Add a legend to differentiate the tracks
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    plt.tight_layout()
    plt.show()


def plotly_4_paths(filepath='erikas_1min.csv', frame_start=500, frame_end=750):
    """
    use plotly to plot the paths of multiple fish (without their skeletons) in polar coordinates.
    note: plot_4_paths does the same thing but with matplotlib (but is less pretty)

    inputs:
    filepath - path to the CSV file containing the data
    frame_start (int): starting frame index for plotting. default is 500.
    frame_end (int): emdomg frame index for plotting. default is 750.
    """

    df = load_and_preprocess_data(filepath=filepath, center_method='least_squares')
    max_radius = df[['mouth_radius', 'L_eye_radius', 'R_eye_radius', 'tail_radius', 'spine_radius']].max().max()

    fig1 = go.Figure()

    # loop to plot each track
    for i, track_id in enumerate(track_ids):
        df_track = df[(df['track'] == track_id) & (df['frame_idx'] >= frame_start) & (df['frame_idx'] <= frame_end)]
        df_track = df_track.sort_values('frame_idx')  # Ensure the DataFrame is sorted by frame index

        # adjust the theta for Plotly (convert radians to degrees and adjust for polar orientation)
        adjusted_theta = -(np.rad2deg(df_track['spine_theta']) + 270) % 360  # this is the correct conversion for plotly conventions

        # Adding the trace for each track to the figure
        fig1.add_trace(go.Scatterpolar(
            r=df_track['spine_radius'],
            theta=adjusted_theta,
            mode='lines',
            name=track_id,
            line=dict(color=px_standard_colors[i], dash='dashdot', width=2),
            marker=dict(color=px_standard_colors[i], size=5, line=dict(width=2))
        ))

        # add circle marker at the start of each track
        if not df_track.empty:
            fig1.add_trace(go.Scatterpolar(
                r=[df_track.iloc[0]['spine_radius']],
                theta=[-(np.rad2deg(df_track.iloc[0]['spine_theta']) + 270) % 360],
                mode='markers',
                marker=dict(color=px_standard_colors[i], symbol='circle', size=8),
                showlegend=False
            ))

        # add triangle marker at the end of each track
        fig1.add_trace(go.Scatterpolar(
            r=[df_track.iloc[-1]['spine_radius']],
            theta=[-(np.rad2deg(df_track.iloc[-1]['spine_theta']) + 270) % 360],
            mode='markers',
            marker=dict(color=px_standard_colors[i], symbol='triangle-left', size=12),
            showlegend=False
        ))

    # adjustments
    fig1.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max_radius]),
            angularaxis=dict(direction='clockwise', thetaunit='degrees')
        ),
        title=f'Paths for Multiple Fish from Frame {frame_start} to {frame_end}',
        width=1000,
        height=500,
    )

    fig1.update_layout(
        title={
            'text': f"Paths for the Fish from Frame {frame_start} to {frame_end}<br><sub><i>hover over the path to display polar coordinates + travel with the fish    : ) </i> </sub>",
            'y':0.9,
            'x':0.05,
            'xanchor': 'left',
            'yanchor': 'top'
        }
    )

    fig1.show()
    return fig1

# relative size plots
def plotly_relative_sizes(summary_df, write_html=False):
    """
    Generate a bar plot of relative fish sizes using Plotly.

    Parameters:
    - summary_df: DataFrame containing summary statistics of fish sizes.
    """

    title="Relative Fish Size"
    subtitle="as computed by average skeleton lengths across frames"

    # Create the bar plot
    fig = px.bar(summary_df, y='Track', x='Average Length', error_x='Standard Deviation',
                 width=1000,
                 height=400,
                 title=title,  # This will be overridden
                 labels={'Average Length': 'Average Skeleton Length in Pixels', 'Track': 'Track'},
                 hover_data={'Relative Size (%)': True, 'Count': True},
                 color='Track',  # Ensure color is based on track
                 color_discrete_sequence=px_standard_colors,
                 orientation='h')  # 'h' for horizontal orientation

    # Update the layout to include main title and subtitle
    fig.update_layout(
        title={
            'text': f"{title}<br><sub><i>{subtitle}</i></sub>",
            'y':0.9,
            'x':0.05,
            'xanchor': 'left',
            'yanchor': 'top'
        }
    )

    # Adjust other aspects of the traces
    fig.update_traces(marker=dict(opacity=0.75), width=0.6)

    # Display the figure
    fig.show()

    if write_html:
        fig.write_html(f"relative_fish_sizes.html")
    # return fig      # since figures can be written to html


# add this
def size_over_time():
  pass

# velocity and acceleration plots

def plot_velocity_profiles(filepath='erikas_1min.csv', center_method='least_squares', plot_type='plain', win=25, poly=3, figure_title='Velocity Profiles', xlim=(-5, 50), figsize=(14, 4)):
    """
    use matplotlib to plot horizontal violin plots for either plain or smoothed velocity data calculated for each track.

    filepath: Path to the CSV file containing the data.
    center_method: Method used for centering (default 'least_squares').
    plot_type: 'plain' for raw velocities or 'smooth' for smoothed velocities.
    figure_title: Title for the entire figure.
    xlim: Tuple for x-axis limits.
    figsize: Tuple for figure size.
    """
    df = load_and_preprocess_data(filepath=filepath, center_method=center_method)

    # Prepare velocity dataframes depending on the plot type
    if plot_type == 'plain':
        velocity_dfs = [calc_velocity_plain(df, f'track_{i}') for i in range(4)]
        titles = [f'Velocity Magnitude for Track {i}' for i in range(4)]
    elif plot_type == 'smooth':
        velocity_dfs = [calc_velocity_smooth(df, f'track_{i}', win=win, poly=poly) for i in range(4)]
        titles = [f'Smoothed Velocity for Track {i}' for i in range(4)]
    else:
        raise ValueError("Invalid plot_type. Use 'plain' or 'smooth'.")


    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, sharex=True)

    for i, ax in enumerate(axes.flatten()):
        sns.violinplot(x=velocity_dfs[i]['velocity'], ax=ax, orient='h', inner='quartile', fill=True, saturation=1, color=standard_colors[i])
        ax.set_title(titles[i], fontsize=10)
        ax.set_ylabel(f'Track {i}')
        ax.set_xlim(xlim)

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.85)
    fig.suptitle(figure_title, fontsize=12)
    plt.show()


def plotly_velocity_profiles(filepath='erikas_1min.csv', center_method='least_squares', plot_type='plain', win=25, poly=3, figure_title='Velocity Profiles', yaxis_range=[-5,30]):
    """
    use plotly to plot violin plots for either plain or smoothed velocity data calculated for each track

    filepath: Path to the CSV file containing the data.
    center_method: Method used for centering (default 'least_squares').
    plot_type: 'plain' for raw velocities or 'smooth' for smoothed velocities (win=25, poly=3 by default)
    figure_title: title for the entire figure.
    """
    df = load_and_preprocess_data(filepath=filepath, center_method=center_method)

    # prepare velocity dataframes depending on the plot type
    if plot_type == 'plain':
        velocity_dfs = [calc_velocity_plain(df, f'track_{i}') for i in range(4)]
    elif plot_type == 'smooth':
        velocity_dfs = [calc_velocity_smooth(df, f'track_{i}', win=win, poly=poly) for i in range(4)]
    else:
        raise ValueError("Invalid plot_type. Use 'plain' or 'smooth'.")

    fig = go.Figure()

    # add violin plots for each track
    for i, vel_df in enumerate(velocity_dfs):
        fig.add_trace(go.Violin(y=vel_df['velocity'],
                                name=f'Track {i}',
                                box_visible=True,
                                meanline_visible=True,
                                line_color=px_standard_colors[i],
                                # fillcolor=px_standard_colors[i],
                                opacity=1))

    # update layout to suit your preferences
    fig.update_layout(
        title=figure_title,
        xaxis_title='Track',
        yaxis_title='Velocity',
        yaxis_range=yaxis_range,
        violingap=0.3,
        violinmode='overlay'  # Overlay to compare distributions directly
    )

    fig.show()



def plotly_velocity_profiles_clean(filepath='erikas_1min.csv', center_method='least_squares', win=25, poly=3, figure_title='Velocity Profiles', save_image=False):
    """
    plot both plain and smoothed velocity distributions side by side for each track using Plotly

    filepath: path to the csv containing the data.
    center_method: Method used for centering (default 'least_squares').
    win: window size for the smoothing filter.
    poly: polyn degree for the smoothing filter.
    figure_title: Title for the entire figure.
    """
    df = load_and_preprocess_data(filepath=filepath, center_method=center_method)

    # prep plain and smoothed velocity dataframes
    velocity_dfs_plain = [calc_velocity_plain(df, f'track_{i}') for i in range(4)]
    velocity_dfs_smooth = [calc_velocity_smooth(df, f'track_{i}', win=win, poly=poly) for i in range(4)]

    fig = go.Figure()
    colors = ['deepskyblue', 'salmon', 'tan', 'violet']
    for i in range(4):
        # group plain and smoothed velocities together by adjusting their positions slightly
        group_position = f'Track {i}'

        # add plain velocity violin
        fig.add_trace(go.Violin(y=velocity_dfs_plain[i]['velocity'],
                                name=f'Plain',
                                legendgroup=f'Track {i}',
                                scalegroup=f'Track {i}',
                                line_color=px_standard_colors[i],
                                box_visible=True,
                                meanline_visible=True,
                                opacity=1,
                                side='negative',
                                x0=group_position,
                                width=0.4,
                                points='all',
                                jitter=0.15,

                                ))

        # add smoothed velocity violin
        fig.add_trace(go.Violin(y=velocity_dfs_smooth[i]['velocity'],
                                name=f'Smooth',
                                legendgroup=f'Track {i}',
                                scalegroup=f'Track {i}',
                                line_color=colors[i],
                                box_visible=True,
                                meanline_visible=True,
                                opacity=0.9,
                                side='positive',
                                x0=group_position,
                                width=0.4,
                                points='all',
                                jitter=0.15,
                                ))


    fig.update_layout(
        title=figure_title,
        violinmode='group',
        showlegend=True,
        xaxis=dict(showticklabels=True),
        yaxis_title='Velocity',
        # template='plotly_white',
        violingap=0.4
    )

# high-resolution settings
#     high_res_width = 2400  # Width in pixels for high resolution
#     high_res_height = 1200  # Height in pixels for high resolution

    if save_image:
        fig.write_image('highres.png', scale=6)  # scale parameter to enhance the resolution
    fig.show()



def plot_velocity_ts(ax, data, track_name, smoothed, color='purple', alpha=0.5, linestyle='-'):
    """
    plot the velocity (as a time series) of a specified track from a df on whatever axes you choose

    note: make sure to have the line
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    right before calling this function

    my inputs:
        ax: axes on which to plot (allows for us to overlay multiple plots)
        df: df w/ velocity data from either calc_velocity_plain or calc_velocity_smooth
        track_name (str): label for the plot legend.
        smoothed (bool): write whether the data is smoothed.
        color (str): color of plot line
        alpha (float): transparency of the plot line
        linestyle (str): style of the plot line
    """
    # this is already sorted by frame index but let's be safe
    data_sorted = data.sort_values('frame_idx', ascending=True)

    # extract x and y values for plotting purposes
    xs = data_sorted['frame_idx']
    ys = data_sorted['velocity']

    # plot on the provided axes
    ax.plot(xs, ys, label=f"{track_name} {'Smoothed' if smoothed else 'Raw'}", color=color, linewidth=1.5, alpha=alpha, linestyle=linestyle)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Velocity')
    # ax.set_ylim([0, 50])
    ax.legend()


def plot_acceleration_ts(ax, data, track_name, color='green', alpha=0.5, linestyle='-'):
    """
    plot the acceleration of a specified track from a dataframe on whatever axes you choose

    note: make sure to have the line
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    right before calling this function

    my inputs:
        ax: axes on which to plot (allows for us to overlay multiple plots)
        df: df w/ acceleration data from calc_acceleration_smooth or another acceleration funct
        track_name (str): label for the plot legend.
        color (str): color of plot line
        alpha (float): transparency of the plot line
        linestyle (str): style of the plot line
    """
    # this is already sorted by frame index but let's be safe
    data_sorted = data.sort_values('frame_idx', ascending=True)

    # extract x and y values for plotting purposes
    xs = data_sorted['frame_idx']
    ys = data_sorted['acceleration']

    # plot on the provided axes
    ax.plot(xs, ys, label=f"{track_name} {'Smoothed acceleration'}", color=color, linewidth=1, alpha=alpha, linestyle=linestyle)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Velocity')
    ax.legend()


def plotly_velocity_slide(filepath='erikas_1min.csv', center_method='least_squares', win=25, y_range=(0,50)):
    """
    Plot velocities over time with an interactive slider to adjust the smoothing window size.

    filepath: Path to the CSV file containing the data.
    center_method: Method used for centering (default 'least_squares').
    win: Initial window size for the smoothing filter.
    y_range: Tuple specifying the y-axis range.
    """
    df = load_and_preprocess_data(filepath=filepath, center_method=center_method)

    fig = go.Figure()

    # create traces for each track with initial smoothing
    traces = []
    for i in range(4):
        track_id = f'track_{i}'
        velocities = calc_velocity_smooth(df, track_id, win=win, poly=3)
        trace = go.Scatter(
            visible=True,
            line=dict(color=px_standard_colors[i], dash='solid', width=4),
            opacity=0.65,
            name=f"Track {i}",
            x=velocities['frame_idx'],
            y=velocities['velocity'],
            fill='tozeroy',  # Fill to the y=0 line
            mode='lines'
        )
        fig.add_trace(trace)
        traces.append(trace)

    # define steps for the slider, each step updates all traces
    winsizes = []
    for winsize in np.arange(5, 115, 2):  # smoothing window from 5 to 501, steps of 2
        winsize_dict = {
            'method': 'restyle',
            'args': [{'y': [calc_velocity_smooth(df, f'track_{i}', win=winsize, poly=3)['velocity'].tolist() for i in range(4)]}],
            'label': str(winsize)
        }
        winsizes.append(winsize_dict)

    # create and add slider
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Window Size: "},
        pad={"t": 50},
        steps=winsizes
    )]

    # Update layout to include slider and fix y-axis range
    fig.update_layout(
        sliders=sliders,
        title="Effect of Savitzsky Golay Window Size on Velocity",
        xaxis_title="Frame Index",
        yaxis_title="Velocity",
        yaxis=dict(fixedrange=True, range=y_range),  # Fix the y-axis range
        # plot_bgcolor='white'
        template='plotly_white'
    )

    fig.show()


# wavelet visuals
def visualize_activity_and_wavelet(velo_df, activity_results, track, window_size, scales, y_max=60):
    """
    visualizes velocity data, activity levels, and wavelet coefficients
    """
    wavelet_energy = activity_results[track]['energy']
    coefficients = activity_results[track]['coefficients']

    # create a fig with two subplots (shared x-axis)
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    # top plot: velocity and activity levels
    ax0 = plt.subplot(gs[0])
    ax0.plot(velo_df.index[:len(wavelet_energy)*window_size], velo_df[track][:len(wavelet_energy)*window_size], label='Velocity', color='black')
    ax0.set_title(f'Velocity and Activity Levels for {track}')

    # overlaying the activity levels
    for label, color in zip(['idle', 'slightly active', 'moderately active', 'very active'], ['lightblue', 'green', 'yellow', 'red']):
        mask = wavelet_energy['activity_level'] == label
        mask_repeated = np.repeat(mask.values, window_size)[:len(velo_df[track][:len(wavelet_energy)*window_size])]

        ax0.fill_between(velo_df.index[:len(wavelet_energy)*window_size],
                            0,
                            y_max,
                            where=mask_repeated, alpha=0.3, color=color, label=label)

    ax0.set_ylim(0, y_max)
    ax0.set_xlabel('Frame Index')
    ax0.set_ylabel('Velocity')
    ax0.legend()

    # second plot: wavelet coefficients
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.imshow(np.abs(coefficients), extent=[velo_df.index.min(), velo_df.index.max(), scales.min(), scales.max()],
                cmap='coolwarm', aspect='auto', origin='lower')
    ax1.set_xlabel('Time (Frame Index)')
    ax1.set_ylabel('Scale (Frequency Inverse)')
    ax1.set_title(f'Wavelet Transform of Velocity Data for {track}')

    plt.tight_layout()
    plt.show()
