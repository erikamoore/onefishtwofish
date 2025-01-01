# -*- coding: utf-8 -*-
from constants import *

# checks
import sys
print(sys.executable)
import pandas as pd
print(pd.__version__)

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean, cdist
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
import seaborn as sns
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import matplotlib.patches as patches
from datetime import timedelta
from tqdm.notebook import tqdm as tqdm
from IPython.display import HTML
from plotly.subplots import make_subplots
import plotly.express as px
from fastdtw import fastdtw
from tabulate import tabulate
from matplotlib import cm
from matplotlib.colors import Normalize
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.cluster import KMeans
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pywt
import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")



# ==================== preprocessing and helper functions ========================

# coordinate conversions for preprocessing
def cartesian_to_polar(x, y):
    """
    helper function for adjust_and_convert_coordinates
    converts default sleap 'cartesian' coordinates to polar coordinates with respect to the new origin
    note: returning -theta was necessary due to matplotlib vs SLEAP axes conventions
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, -theta

def polar_to_cartesian(radius, theta):
    """
    converts polar coordinates to cartesian coordinates
    essentially 'undoes' the cartesian_to_polar function
    """
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y



def adjust_and_convert_coordinates(df, center_x, center_y):
    """
    adjusts coordinates to tank center and converts to polar coordinates
    """
    for part in ['mouth', 'L_eye', 'R_eye', 'tail', 'spine']:
        df[f'{part}_adj_x'] = df[f'{part}.x'] - center_x
        df[f'{part}_adj_y'] = df[f'{part}.y'] - center_y
        df[f'{part}_radius'], df[f'{part}_theta'] = zip(*df.apply(
            lambda row: cartesian_to_polar(row[f'{part}_adj_x'], row[f'{part}_adj_y']), axis=1
        ))
    return df


def interpolate_missing_frames(df, interpolation_method='linear'):
    """
    interpolates missing frames for each track in the dataframe and handles missing values.
    note: so far stats on the interpolated data are close with raw data (indicating minimal perturbation).

    parameters:
        df: df containing tracking data with possible missing frames.
        interpolation_method (str): interpolation method to use ('linear', 'quadratic', etc.).

    return:
        df: df with interpolated positions for missing frames and forward/backward
            filled values for numeric columns.
    """
    interpolated_tracks = []

    # look at tracks one at a time
    for track, group in df.groupby('track'):

        # make sure chronological order of frames
        # need all integers between min and max frame num
        group = group.sort_values('frame_idx')
        full_frame_range = np.arange(group['frame_idx'].min(), group['frame_idx'].max() + 1)

        # reindex to include all frame nums, leave NaNs where data is missing
        group_full = group.set_index('frame_idx').reindex(full_frame_range).reset_index()

        # fill our 'track' column with the current track identifier
        group_full['track'] = track

        # we separate numeric and non-numeric columns bc we handle them differently
        numeric_cols = group_full.select_dtypes(include=[np.number]).columns
        non_numeric_cols = group_full.select_dtypes(exclude=[np.number]).columns

        # now, we interpolate numeric columns
        group_full[numeric_cols] = group_full[numeric_cols].interpolate(method=interpolation_method, limit_direction='both')

        # forward fill and backward fill non-numeric columns (basically just the track column)
        group_full[non_numeric_cols] = group_full[non_numeric_cols].ffill().bfill()

        # append the result to our list
        interpolated_tracks.append(group_full)

    # concatenate all interpolated tracks into a final df
    interpolated_df = pd.concat(interpolated_tracks, ignore_index=True)

    return interpolated_df


def load_and_preprocess_data(filepath='erikas_1min.csv', center_method='least_squares', interpolate=True):
    """
    loads the SLEAP data and applies preprocessing steps to adjust and convert coordinates

    inputs:
        filepath: path to the csv
        center_method: either 'midpoint' or 'least_squares'
        interpolate: defaults to True

    returns:
        df: preprocessed df with recentered coordinates (in polar form) with interpolated missing frames
    """
    df = pd.read_csv(filepath)

    # adjusts coordinates as previously defined
    center_x, center_y = (MID_CX, MID_CY) if center_method == 'midpoint' else (LS_CX, LS_CY)
    df = adjust_and_convert_coordinates(df, center_x, center_y)

    if interpolate:
        df = interpolate_missing_frames(df)

    return df


# find missing tracks
def print_missing_tracks(df):
    """
    print the count of frames per track and lists missing frames for each track
    input:
            df containing the adjusted polar coordinates (a preprocessed df)
    returns:
            nothing, but prints the count of frames per track and lists missing frames for each track
    """
    frame_count_per_track = df.groupby('track')['frame_idx'].nunique()
    for track, count in frame_count_per_track.items():
        print(f"{track} was detected in {count} frames.")

    all_frames = df['frame_idx'].unique()
    missing_frames = {}

    for track in df['track'].unique():
        track_frames = df[df['track'] == track]['frame_idx'].unique()
        missing_frames[track] = sorted(set(all_frames) - set(track_frames))

    print()

    for track, frames in missing_frames.items():
        print(f"Missing frames for {track}: {frames}")


# time conversions
def frame_2_time(frame_idx, FPS=30):
    """
    convert a frame index to a corresponding time in minutes and seconds.

    inputs:
    frame_idx (int): The frame index.
    FPS (int): Frames per second, default is 30.

    returns:
    str: Time in the format "minutes:seconds".
    """
    total_seconds = frame_idx / FPS
    time = timedelta(seconds=total_seconds)
    minutes, seconds = divmod(time.seconds, 60)
    return f"{minutes} minutes, {seconds} seconds"


def time_2_frame(minutes, seconds, FPS=30):
    """
    convert a time in minutes and seconds to a corresponding frame index.

    inputs:
    minutes (int): The number of minutes.
    seconds (int): The number of seconds.
    FPS (int): Frames per second, default is 30.

    return:
    int: The corresponding frame index.
    """
    total_seconds = minutes * 60 + seconds
    return int(total_seconds * FPS)


def sec_2_frame(seconds, FPS=30):
    """
    convert a time in seconds to a corresponding frame index.

    input:
    seconds (int): The number of seconds.
    FPS (int): Frames per second, default is 30.

    returns:
    int: The corresponding frame index.
    """
    return int(seconds * FPS)


# fish size helper
def calculate_edge_length(p1, p2):
    """
    calculates the length of an edge between two nodes
    uses np.linalg.norm to calculate the euclidean distance between two points

    used as a helper function to estimate relative fish size
    """
    return np.linalg.norm(np.array(p2) - np.array(p1))

# def calculate_edge_length(coord1, coord2):
#     return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

# color helper functions
def lighten_color(color, amount=0.2):
    """ lighten a given color by increasing its brightness."""
    color = color.replace('rgb', '').replace('(', '').replace(')', '').split(',')
    r, g, b = [int(x) for x in color]
    r = int(min(255, r + r * amount))
    g = int(min(255, g + g * amount))
    b = int(min(255, b + b * amount))
    return f'rgb({r}, {g}, {b})'

def set_opacity(color, opacity=0.3):
    """ set the opacity for a given color."""
    # Ensure the color is in RGB format, then convert it to RGBA with the specified opacity.
    color = color.replace('rgb', '').replace('(', '').replace(')', '').split(',')
    r, g, b = [int(x) for x in color]
    return f'rgba({r}, {g}, {b}, {opacity})'


def combine_plotly_figs_to_html(plotly_figs, html_fname, include_plotlyjs='cdn',
                                separator=None, auto_open=False):
    """
    combines multiple plotly figures into a single html file
    """
    with open(html_fname, 'w') as f:
        f.write(plotly_figs[0].to_html(include_plotlyjs=include_plotlyjs))
        for fig in plotly_figs[1:]:
            if separator:
                f.write(separator)
            f.write(fig.to_html(full_html=False, include_plotlyjs=False))

    if auto_open:
        import pathlib, webbrowser
        uri = pathlib.Path(html_fname).absolute().as_uri()
        webbrowser.open(uri)

# ==================== plain vanilla distance functions (avg dist to center, dist between pairs) ======================


# distances from center
def calc_avg_center_dist(df):
    """
    calculates the average distance to the center for each track

    input:
    df containing the adjusted polar coordinates (a preprocessed df)

    returns:
    a dictionary with track IDs as keys and average distances as values.
    """
    avg_distances = df.groupby('track')['spine_radius'].mean().reset_index()
    avg_distances_dict = avg_distances.set_index('track')['spine_radius'].to_dict()
    return avg_distances_dict


def calc_dist_for_frame(df, frame_idx, track_id):
    """
    was mainly using this as a simple way to help check my work for the nearest neighbors distance funct...
    calculates distances from a specified track to all other tracks within a specific frame.

    inputs:
    df containing the adjusted polar coordinates (a preprocessed df)
    frame_idx (int): The specific frame index to filter by.
    track_id (str): The track identifier to calculate distances from.

    returns:
    distances (dict): A dictionary containing distances from the specified track to all other tracks.
    """
    df_frame = df[df['frame_idx'] == frame_idx]
    df_track = df_frame[df_frame['track'] == track_id]
    if df_track.empty:
        return {}

    x1, y1 = df_track[['spine_adj_x', 'spine_adj_y']].values[0]
    distances = {}

    for other_track_id in df_frame['track'].unique():
        if other_track_id != track_id:
            df_other = df_frame[df_frame['track'] == other_track_id]
            x2, y2 = df_other[['spine_adj_x', 'spine_adj_y']].values[0]
            distance = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
            distances[other_track_id] = distance
            print(f"Track is {track_id}, distance to {other_track_id} is {distance:.2f}")

    return distances



def calc_dist_for_pair(df, track1, track2, window_size=50):
    """
    inputs:
        df containing the adjusted coordinates (a preprocessed df)
        track1 (str): the first fish track (instance) to filter by.
        track2 (str): the second fish track (instance) to filter by.

    returns:
        result_df: df containing the frame indices and calculated distance
                    between the specified tracks for every frame, includes rolling avg
    """

    # filter df for desired tracks
    df_track1 = df[df['track'] == track1]
    df_track2 = df[df['track'] == track2]

    # ensure we match frames
    matched_frames = pd.merge(df_track1, df_track2, on='frame_idx', suffixes=('_1', '_2'))

    # extract coordinates
    x1 = matched_frames['spine_adj_x_1'].values
    y1 = matched_frames['spine_adj_y_1'].values
    x2 = matched_frames['spine_adj_x_2'].values
    y2 = matched_frames['spine_adj_y_2'].values

    # calc dist
    # distances = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    distances = np.linalg.norm(np.vstack((x2 - x1, y2 - y1)).T, axis=1) # slightly more efficient but same as line above


    # create result df
    result_df = pd.DataFrame({
        'frame_idx': matched_frames['frame_idx'],
        'track1': track1,
        'track2': track2,
        'distance': distances
    })

    # calc rolling avg of dist
    # 'min_periods=1' to ensure the calculation is done even with fewer items than window size
    result_df['rolling_avg_distance'] = result_df['distance'].rolling(window=window_size, min_periods=1, center=True).mean()


    return result_df



def calculate_all_pairs_distances(df, track_ids):
    """
    calculate the distances between all pairs for all frames

    inputs:
        df containing the adjusted polar coordinates (a preprocessed df)
        track_ids (list): a list of track IDs to calculate distances for.

    returns:
        a df containing the distances between all pairs for all frames
    """
    pairs_distances = []
    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            track1, track2 = track_ids[i], track_ids[j]
            distances = calc_dist_for_pair(df, track1, track2)
            distances['pair'] = f"{track1}-{track2}"
            pairs_distances.append(distances)
    dist_df = pd.concat(pairs_distances)
    pivoted_dist_df = dist_df.pivot(index='frame_idx', columns='pair', values='distance')
    return pivoted_dist_df

# ============================ nearest neighbor distance functions ======================================

def calculate_nearest_neighbor_distances(df, track_ids, frame_range=(0, None)):
    """
    calculates the distance of each track to its nearest neighbor within a specific frame range
    defaults to all frames if no frame_range is specified

    returns two dataframes:
        the first is the distances of each track to its nearest neighbor
        the second is the track_id of the nearest neighbor of each track
    """
    df = df.sort_values('frame_idx')  # just to be safe

    # filter for frame range
    if frame_range[1] is not None:
        df = df[(df['frame_idx'] >= frame_range[0]) & (df['frame_idx'] <= frame_range[1])]
    else:
        df = df[df['frame_idx'] >= frame_range[0]]

    # dfs to store nearest neighbor distances and corresponding track IDs
    nnd_distances_df = pd.DataFrame(index=df['frame_idx'].unique())
    nnd_track_ids_df = pd.DataFrame(index=df['frame_idx'].unique())

    # create empty columns for each track_id
    for track_id in track_ids:
        nnd_distances_df[f'nnd_{track_id}'] = np.nan
        nnd_track_ids_df[f'nn_of_{track_id}'] = None

    # loop through the frames
    # for frame in df['frame_idx'].unique():
    for frame in tqdm(df['frame_idx'].unique(), desc='Processing frames...'):
        frame_data = df[df['frame_idx'] == frame]

        # create a dictionary to hold the positions of each track
        track_positions = {track_id: frame_data[frame_data['track'] == track_id] for track_id in track_ids}

        # loop through the tracks
        for track_id in track_ids:
            current_fish = track_positions[track_id]
            if current_fish.empty:
                continue

            other_fish = frame_data[frame_data['track'] != track_id]
            if other_fish.empty:
                continue

            # vectorized distance computation using np.linalg.norm (just the dist formula)
            distances = np.linalg.norm(other_fish[['spine_adj_x', 'spine_adj_y']].values - current_fish[['spine_adj_x', 'spine_adj_y']].values, axis=1)

            min_distance_idx = distances.argmin()
            nearest_neighbor_distance = distances[min_distance_idx]
            nearest_neighbor_track = other_fish.iloc[min_distance_idx]['track']

            nnd_distances_df.loc[frame, f'nnd_{track_id}'] = nearest_neighbor_distance
            nnd_track_ids_df.loc[frame, f'nn_of_{track_id}'] = nearest_neighbor_track

            # debug print statement to check nearest neighbor consistency
            # print(f"Frame {frame}, Track {track_id}: Nearest Neighbor is {nearest_neighbor_track} with distance {nearest_neighbor_distance}")

    return nnd_distances_df, nnd_track_ids_df




def calculate_nnd_avg_stats(nnd_distances_df, window_size=30):
    """
    calc the average and moving average nearest neighbor distances for each track.

    inputs:
    nnd_distances_df (pd.DataFrame): df containing nearest neighbor distances.
    window_size (int):  window size for calculating the moving average.

    return:
    tuple: average nearest neighbor distances (series) and moving average nearest neighbor distances (df).
    """
    avg_nnd = nnd_distances_df.mean()
    moving_avg_nnd = nnd_distances_df.rolling(window=window_size, min_periods=1).mean()
    return avg_nnd, moving_avg_nnd



def get_most_common_nearest_neighbors(nnd_track_ids_df, track_ids):
    """
    intended to be called following calculate_nearest_neighbor_distances

    inputs:
    the nnd_distances_df (nearest neighbor distances) returned by calculate_nearest_neighbor_distances
    track_ids (list): a list of track IDs

    returns:
    two dicts:
        most_common_nns: a dictionary mapping track IDs to the most common nearest neighbor for that track

        neigh_tallies: a dictionary mapping track IDs to their respective nearest neighbor tallies
                        (the number of times every other track was a nearest neighbor of that track)
    """
    most_common_nns = {}
    neigh_tallies = {}
    for track_id in track_ids:
        neigh_counts = nnd_track_ids_df[f'nn_of_{track_id}'].value_counts()
        # print(f"neigh_counts for {track_id}:\n{neigh_counts}\n")
        neigh_tallies[track_id] = neigh_counts
        most_common_nns[track_id] = neigh_counts.idxmax()

    return most_common_nns, neigh_tallies


# return here

def rank_nearest_neighbors(neigh_tallies):
    """
    ranks tracks based on how frequently they were a nearest neighbor of other tracks
    intended to be called following get_most_common_nearest_neighbors (which returns neigh_tallies as its second output)

    inputs:
    neigh_tallies (dict): a dictionary mapping track IDs to their respective nearest neighbor tallies
                            (the number of times every other track was a nearest neighbor of that track)

    returns:
    pd.Series: Series of ranked neighbors by popularity.
    """
    total_tallies = pd.Series(dtype=int)
    for tally in neigh_tallies.values():
        total_tallies = total_tallies.add(tally, fill_value=0)
    return total_tallies.sort_values(ascending=False)




def analyze_nearest_neighbors(filepath, track_ids, center_method = 'least_squares', frame_range=(0, None)):
    """
    convenience function that handles the entire 'nearest neighbor' workflow

    analyzes the nearest neighbors for a set of tracks in a given data file
    calls functions calculate_nearest_neighbor_distances, get_most_common_nearest_neighbors, and rank_nearest_neighbors

    inputs:
            filepath (str): path to the CSV file containing the data.
            track_ids (list): a list of track IDs to analyze.
            frame_range (tuple): a tuple specifying the frame range to analyze.

    returns:
            nnd_distances_df (df): distances of each track to its nearest neighbor.
            nnd_track_ids_df (df): track IDs of the nearest neighbor of each track.
            most_common_nns (dict): maps track IDs to the most common nearest neighbor for that track.
            neigh_tallies (dict): maps track IDs to their respective nearest neighbor tallies
            ranked_neighbors (series): ranked neighbors by popularity.
            """

    print(f"Loading and preprocessing data...\n")
    df = load_and_preprocess_data(filepath, center_method=center_method)
    print("*"*70)
    print("Calculating nearest neighbor distances and corresponding ids...\n")

    nnd_distances_df, nnd_track_ids_df = calculate_nearest_neighbor_distances(df, track_ids, frame_range)
    print("Nearest Neighbor Distances:")
    print(nnd_distances_df)
    print("\nNearest Neighbor Track IDs:")
    print(nnd_track_ids_df)


    print("*"*70)
    avg_nnd, moving_avg_nnd = calculate_nnd_avg_stats(nnd_distances_df, window_size=30)
    print("Average Nearest Neighbor Distances:")
    for track, distance in avg_nnd.items():
        print(f"{track:}: {distance:>10.2f}")
    print("\nMoving (Rolling) Average Nearest Neighbor Distances:\n",moving_avg_nnd)

    print("*"*70)
    most_common_nns, neigh_tallies = get_most_common_nearest_neighbors(nnd_track_ids_df, track_ids)
    ranked_neighbors = rank_nearest_neighbors(neigh_tallies)
    print(f"\nSummary of Results:")
    print("Most Common Nearest Neighbors:")
    for track_id, nn in most_common_nns.items():
        print(f"The most common nearest neighbor for {track_id} is {nn}.")

    print("\nPopularity Ranking of Nearest Neighbors:")
    for i, (neighbor, count) in enumerate(ranked_neighbors.items(), start=1):
        print(f"{i}. {neighbor} was the nearest neighbor {count} times.")

    most_popular_neighbor = ranked_neighbors.idxmax()
    least_popular_neighbor = ranked_neighbors.idxmin()
    print(f"\nThe most popular neighbor is {most_popular_neighbor}, appearing {ranked_neighbors[most_popular_neighbor]} times.")
    print(f"The least popular neighbor is {least_popular_neighbor}, appearing {ranked_neighbors[least_popular_neighbor]} times.")

    return nnd_distances_df, nnd_track_ids_df, most_common_nns, neigh_tallies, ranked_neighbors

def calculate_nearest_neighbor_distances_optimized(df, track_ids, frame_range=(0, None)):
    """
    Optimized version of nearest neighbor calculation that leverages vectorized operations.

    Parameters:
        df: DataFrame containing interpolated tracking data with adjusted coordinates.
        track_ids: List of track IDs to analyze.
        frame_range: Tuple indicating the frame range to analyze.

    Returns:
        DataFrame with nearest neighbor distances and track IDs.
    """

    # filt for frame range
    if frame_range[1] is not None:
        df = df[(df['frame_idx'] >= frame_range[0]) & (df['frame_idx'] <= frame_range[1])]
    else:
        df = df[df['frame_idx'] >= frame_range[0]]

    nnd_distances = {f'nnd_{track_id}': [] for track_id in track_ids}
    nnd_tracks = {f'nn_of_{track_id}': [] for track_id in track_ids}
    frame_indices = []

    for frame in tqdm(df['frame_idx'].unique(), desc="Processing frames"):
        frame_data = df[df['frame_idx'] == frame]

        # extract coords and calculate pairwise distances in vectorized form
        coordinates = frame_data[['spine_adj_x', 'spine_adj_y']].values
        pairwise_distances = cdist(coordinates, coordinates)
        np.fill_diagonal(pairwise_distances, np.inf)  # ignore dist to self

        # looping through each track and finding its nearest neighbor
        for idx, track_id in enumerate(frame_data['track'].values):
            min_distance_idx = pairwise_distances[idx].argmin()
            nearest_distance = pairwise_distances[idx, min_distance_idx]
            nearest_track = frame_data['track'].values[min_distance_idx]

            # Append results to lists for each track
            nnd_distances[f'nnd_{track_id}'].append(nearest_distance)
            nnd_tracks[f'nn_of_{track_id}'].append(nearest_track)

        frame_indices.append(frame)

    # converting the dictionary of results to dfs
    nnd_distances_df = pd.DataFrame(nnd_distances, index=frame_indices)
    nnd_track_ids_df = pd.DataFrame(nnd_tracks, index=frame_indices)

    return nnd_distances_df, nnd_track_ids_df

# ================================ size estimation functions =======================================


def estimate_fish_sizes(df, parts, edges):
    """
    estimates fish sizes based on a preprocessed df of tracking data
    inputs: df, a preprocessed df of tracking data
    parts: a list of labeled body parts 
    edges: a list of tuples representing connections (edges) between body parts
    """

    # get all of our columns for the body parts
    x_columns = [f"{part}_adj_x" for part in parts]
    y_columns = [f"{part}_adj_y" for part in parts]

    # initialize a list to store the average lengths of all edges for each fish track
    track_lengths = []
    for track_id, track_data in df.groupby('track'):
        # track_data contains all rows for a specified track
        # ex Processing track_id: track_0, track_data shape: (35993, 38) 
        print(f"Processing track_id: {track_id}, track_data shape: {track_data.shape}")
        print(track_data.head())  #  first few rows

        # coords is a dictionary where 
        # keys are body parts for a given track
        # values are arrays of (x,y) coordinates for all frames in current track 

        # here we extract all of the coordinates for each body part 
        coords = {part: track_data[[f"{part}_adj_x", f"{part}_adj_y"]].values for part in parts}
        
        print(f"\ncoords is {coords}")
        for part, values in coords.items():
            print(f"  {part}: {values.shape}, first three rows (i.e., frames): {values[0:3]}")

        print(f"\n track_data length is {len(track_data)}")
        
        # stores summed edge lengths for each frame in current track
        lengths_per_frame = []
        for frame_index in range(len(track_data)):
            # for the current frame, calculate the length of each edge
            edge_lengths = []
            for part1, part2 in edges:
                if part1 in coords and part2 in coords:
                    # calc the distance between the two body parts
                    edge_length = calculate_edge_length(coords[part1][frame_index], coords[part2][frame_index])
                    edge_lengths.append(edge_length)
            
            # summing up the lengths of all edges for the current frame
            total_length = np.sum(edge_lengths)
            lengths_per_frame.append(total_length)

        # more concise, less readable
        # lengths_per_frame = [
        #     np.sum([calculate_edge_length(coords[part1][i], coords[part2][i])
        #             for part1, part2 in edges if part1 in coords and part2 in coords])
        #     for i in range(len(track_data))
        # ]
        
        print(f"  first 5 lengths for track {track_id}: {lengths_per_frame[:5]}")

        # calc the avg for the total edge lengths across all frames for the track
        # keep track of our track_id
        track_lengths.append({'track': track_id, 'average_length': np.mean(lengths_per_frame)})

    summary_df = pd.DataFrame(track_lengths)

    return summary_df



# ================================ dynamic time warping functions ======================================

def calculate_dtw_distance(segment1, segment2):
    """ calculate the Dynamic Time Warping (DTW) distance between two trajectory segments """
    distance, _ = fastdtw(segment1, segment2, dist=euclidean)
    return distance

def compute_dtw_distances(df, window_size=30, step_size=10):
    """
    calculate the average DTW distance between trajectory segments for each track within a sliding window
    and return a list of tuples containing the start frame, end frame, and average DTW distance for each window.

    inputs:
        df (DataFrame): df containing the adjusted polar coordinates (preprocessed)
        window_size (int): the size of the sliding window (how many frames each segment will include)
        step_size (int): num of frames to skip between the start of one window and the start of the next

    returns:
        list: a list of tuples containing the start frame, end frame, and average DTW distance for each window
                sorted in order of ascending distance (i.e. most similar to least)
    """
    print(f"window_size is {window_size}")
    print(f"step_size is {step_size}")
    print(f"*"*75)

    tracks = df['track'].unique()
    dtw_results = []
    total_frames = len(df['frame_idx'].unique()) # used to control loop that processes each window

    print(f"Computing DTW distances: ")
    for start_frame in tqdm(range(0, total_frames - window_size, step_size), unit="win",  # 'win' refers to 'window'
                            bar_format="{percentage:3.0f}%"):
        end_frame = start_frame + window_size
        window_distances = []

        # for each pair of tracks (where i<j to avoid repeating pairs or comparing track to itself)
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                # extract segments of data within current window
                segment1 = df[(df['track'] == tracks[i]) & (df['frame_idx'] >= start_frame) & (df['frame_idx'] < end_frame)]
                segment2 = df[(df['track'] == tracks[j]) & (df['frame_idx'] >= start_frame) & (df['frame_idx'] < end_frame)]

                # check if segments are not empty before calculating distance (might have error if no data present)
                if not segment1.empty and not segment2.empty:
                    distance = calculate_dtw_distance(segment1[['spine_theta', 'spine_radius']].values,
                                                        segment2[['spine_theta', 'spine_radius']].values)

                    # add distance to list of window distances for the current window
                    window_distances.append(distance)

        # after processing all track pairs for the current window, if any distances were computed we take the mean of them
        if window_distances:
            average_distance = np.mean(window_distances)   # average distance for the current window (across all our combinations of tracks)
            # append a tuple with the start frame, end frame, and avg dtw dist of that window to our results list
            dtw_results.append((start_frame, end_frame, average_distance))

    print(f"Complete!")
    print(f"*"*75)
    # print(f"dtw_results is {dtw_results}")
    return dtw_results


def calculate_dtw_durations(dtw_results, lower_percentile=5, upper_percentile=95):
    """
    calculate the total duration of similar paths below a specified DTW distance cutoff percentile

    inputs:
        dtw_results (list): list of tuples containing the start frame, end frame, and DTW distance
                            sorted in order of ascending start frame (returned by compute_dtw_distances)
        lower_percentile (int): the percentile to use for the lower cutoff (default is 5)
        upper_percentile (int): the percentile to use for the upper cutoff (default is 95)

    returns:
        a dictionary containing information for similar and dissimilar paths based on the cutoffs

    """
    # edge case for if no dtw_calculations were performed (or passed in)...
    # we return 0 for total duration, empty list for merged ranges, and 0 for cutoff
    if not dtw_results:
        return None  # to handle case with no data


    # np.percentile calculates cutoff DTW distance at the specified percentile
    # so it finds the DTW distance below (or above) which a certain percentage of the distances fall
    lower_cutoff_value = np.percentile([d[2] for d in dtw_results], lower_percentile)
    upper_cutoff_value = np.percentile([d[2] for d in dtw_results], upper_percentile)

    # filter the results to include only those segments that are below (or above) that distance cutoff
    similar_frames = [(start, end, dist) for start, end, dist in dtw_results if dist <= lower_cutoff_value]
    # print(f"\nsimilar_frames is {similar_frames}")
    dissimilar_frames = [(start, end, dist) for start, end, dist in dtw_results if dist >= upper_cutoff_value]
    # print(f"dissimilar_frames is {dissimilar_frames}\n")

    def merge_ranges(frames):
        # merge overlapping or contiguous time frames (necessary for our step size potentially causing overlap)
        if not frames:
            return []
        frames.sort(key=lambda x: x[0])  # sort by start frame
        merged = [frames[0]]  # initialize with the first frame range
        for start, end, dist in frames[1:]:  # start from the second frame range
            last_start, last_end, _ = merged[-1]  # last range in merged list
            if start <= last_end:  # see if next frame range starts before the current frame range ends (overlap)
                # if overlap, keep current range start frame and see which range interval had a later end frame
                # use the maximum of the two corresponding dtw distances for merged range
                merged[-1] = (last_start, max(last_end, end), max(merged[-1][2], dist))
            else:
                # otherwise (if no overlap), add the current range to the merged list
                merged.append((start, end, dist))
        return merged  # return the merged list of ranges

    merged_similar_ranges = merge_ranges(similar_frames) # find the overlaps for the similar ranges
    merged_dissimilar_ranges = merge_ranges(dissimilar_frames) # find the overlaps for the dissimilar ranges
    total_similar_duration = sum((end - start + 1) / FPS for start, end, _ in merged_similar_ranges) # calculate the total duration of the merged ranges in seconds through simple conversion
    total_dissimilar_duration = sum((end - start + 1) / FPS for start, end, _ in merged_dissimilar_ranges)

    average_similar_dtw = np.mean([dist for _, _, dist in merged_similar_ranges]) if merged_similar_ranges else 0 # compute the average distances for our similar (or dissimilar) frames
    average_dissimilar_dtw = np.mean([dist for _, _, dist in merged_dissimilar_ranges]) if merged_dissimilar_ranges else 0

    # return a dictionary with info about what we consider to pass for similar, dissimilar, and their corresponding percentile cutoff
    return {
        "similar": (total_similar_duration, merged_similar_ranges, lower_cutoff_value, average_similar_dtw),
        "dissimilar": (total_dissimilar_duration, merged_dissimilar_ranges, upper_cutoff_value, average_dissimilar_dtw),
        "percentiles": (lower_percentile, upper_percentile)
    }

def print_durations_and_distances(results, FPS=30):
    """
    mainly a formatting convenience function for the calculate_dtw_durations function
    prints the output of calculate_dtw_durations in a more readable tabular format

    inputs:
        results (dict): the output of calculate_dtw_durations

    returns:
        None
    """
    if not results:
        print("No data available.")
        return

    for key in ["similar", "dissimilar"]:
        info = results[key]
        total_duration, ranges, cutoff_value, average_dtw = info
        percentile = results["percentiles"][0] if key == "similar" else results["percentiles"][1]
        table_data = [{
            "Frame Range": f"{start}-{end}",
            "Start Time (s)": f"{start / FPS:.2f}",
            "End Time (s)": f"{end / FPS:.2f}",
            "Duration (s)": f"{(end - start + 1) / FPS:.2f}",
            "DTW Distance": f"{dist:.2f}"
        } for start, end, dist in ranges]

        print(f"\n{key.capitalize()} Paths ({percentile}th Percentile Cutoff DTW Distance: {cutoff_value:.2f})")
        print(tabulate(table_data, headers="keys", tablefmt="pretty"))
        print(f"Total duration of {key.lower()} paths: {total_duration:.2f} seconds")
        print(f"Average DTW Distance: {average_dtw:.2f}")

# ================================ velocity (and acceleration) functions ======================================


def calc_velocity_plain(df, track):
    """
    calc the instantaneous velocity for a specified track using the differences in consecutive positions.
    this function assumes uniform time intervals between frames and does the following:
            - computes the difference between consecutive x and y coordinates,
            - calculates the magnitude of these differences to determine the velocity
            - returns a DataFrame with frame indices and the corresponding velocities.

    inputs:
        df: df containing preprocessed data.
        track (str): the track identifier to filter by.

    returns:
        df: df containing the frame indices and calculated velocities.
    """
    # filter df for the desired track and ensure it's sorted by frame index
    df_track = df[df['track'] == track].sort_values('frame_idx')

    # extract positions
    positions = df_track[['spine_adj_x', 'spine_adj_y']].values

    # calc differences in positions
    # assume time intervals are uniform and setting the time step to 1
    dx = np.gradient(positions[:, 0])
    dy = np.gradient(positions[:, 1])

    # calc velocity magnitude from differences
    velocities = np.sqrt(dx**2 + dy**2)

    # or in one line we can do
    # velocities = np.linalg.norm(np.gradient(positions, axis=0), axis=1)


    # Create result DataFrame
    result_df = pd.DataFrame({
        'frame_idx': df_track['frame_idx'],
        'velocity': velocities
    })

    return result_df


def calc_all_velocities_plain(df):
    """
    calc instantaneous velocity for all tracks and display their summary statistics.

    inputs:
        df: DataFrame containing tracking data.

    returns:
        two dfs:
        the first contains the instantaneous velocity for each track at a given frame
        the second contains summary stats for the velocities for each track

    """
    all_velocities = pd.DataFrame()

    for track in df['track'].unique():
        # using calc_velocity_plain to calculate velocities for each track
        velocity_df = calc_velocity_plain(df, track)
        velocity_df['track'] = track  
        all_velocities = pd.concat([all_velocities, velocity_df], ignore_index=True)

    # then we group by track and calculate stats for velocities
    velocity_stats = all_velocities.groupby('track')['velocity'].describe()
    pivoted_df = all_velocities.pivot(index='frame_idx', columns='track', values='velocity')

    return pivoted_df, velocity_stats


def calc_velocity_smooth(df, track, win=25, poly=3):
    """
    calc smoothed velocities for a specified track using a Savitzky-Golay filter.

    inputs:
        df: DataFrame containing tracking data.
        track (str): The track identifier to filter by.
        win (int): The window size for the Savitzky-Golay filter, must be odd.
        poly (int): The polynomial order for the Savitzky-Golay filter.

    returns:
        df: DataFrame with the smoothed velocities.
    """
    # filtering df for the specified track and ensure sorted by frame_idx
    df_track = df[df['track'] == track].sort_values('frame_idx')
    if win % 2 == 0 or win <= poly:
        raise ValueError("Window size must be odd and greater than polynomial degree.")

    # extracting the node location data (spine coordinates)
    node_loc = df_track[['spine_adj_x', 'spine_adj_y']].values

    # initialize the velocity array and apply the filter
    node_loc_vel = np.zeros_like(node_loc)

    for c in range(node_loc.shape[1]): # looping over cols
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1)

    # calc the magnitude of the velocity vector
    node_vel = np.linalg.norm(node_loc_vel, axis=1)

    result_df = pd.DataFrame({
        'frame_idx': df_track['frame_idx'],
        'velocity': node_vel
    })

    return result_df


def calc_all_velocities_smooth(df, win=25, poly=3):
    """
    calc smoothed velocities for all tracks using a Savitzky-Golay filter and display their summary statistics.

    inputs:
        df: DataFrame containing tracking data.
        win (int): The window size for the Savitzky-Golay filter, must be odd.
        poly (int): The polynomial order for the Savitzky-Golay filter.

    returns:
        two dfs:
        the first contains the instantaneous velocity for each track at a given frame
        the second contains summary stats for the velocities for each track

    """
    if win % 2 == 0 or win <= poly:
        raise ValueError("Window size must be odd and greater than polynomial degree.")

    all_velocities = pd.DataFrame()

    for track in df['track'].unique():
        # using calc_smoothed_velocity to calculate velocities for each track
        velocity_df = calc_velocity_smooth(df, track, win, poly)

        velocity_df['track'] = track # grouping by track

        # appending to our all_velocities DataFrame
        all_velocities = pd.concat([all_velocities, velocity_df], ignore_index=True)

    # grouping by each track and calc descriptive statistics for velocities
    velocity_stats = all_velocities.groupby('track')['velocity'].describe()
    pivoted_df = all_velocities.pivot(index='frame_idx', columns='track', values='velocity')

    return pivoted_df, velocity_stats


def calc_acceleration_plain(df, track):
    """
    calc the instantaneous acceleration for a specified track using the differences in consecutive velocities.
    This function assumes uniform time intervals between frames and calculates the differences in consecutive velocities.

    df: df containing preprocessed data
    track: track identifier to filter by.
    returns: df containing the frame indices and calculated accelerations.
    """
    # filter df for the desired track and ensure it's sorted by frame index
    df_track = df[df['track'] == track].sort_values('frame_idx')

    # calc velocities if not already present
    if 'velocity' not in df_track.columns:
        df_track = calc_velocity_plain(df_track, track)

    # calc differences in velocities (assuming time intervals are uniform and setting the time step to 1)
    velocities = df_track['velocity'].values
    accelerations = np.gradient(velocities)

    # create result DataFrame
    result_df = pd.DataFrame({
        'frame_idx': df_track['frame_idx'],
        'acceleration': accelerations
    })

    return result_df



def calc_acceleration_smooth(df, track, win=25, poly=3):
    """
    calculates smoothed accelerations for a specified track using a Savitzky-Golay filter.

    inputs:
        df: df containing the preprocessed
        track (str): The track identifier to filter by.
        win (int): The window size for the Savitzky-Golay filter, must be odd.
        poly (int): The polynomial order for the Savitzky-Golay filter.

    return:
        df with the smoothed accelerations.
    """
    # filter DataFrame for the specified track
    df_track = df[df['track'] == track].sort_values('frame_idx')

    # ensure the window size is odd and greater than the polynomial degree
    if win % 2 == 0 or win <= poly:
        raise ValueError("win must be odd and greater than poly")

    # \oint location data (spine coordinates)
    node_loc = df_track[['spine_adj_x', 'spine_adj_y']].values

    # create the acceleration array
    node_loc_acc = np.zeros_like(node_loc)

    # apply the Savitzky-Golay filter to compute the acceleration for each column (either the x or y coordinate)
    # deriv=2 for calculating the second derivative, which represents acceleration
    for c in range(node_loc.shape[1]):
        node_loc_acc[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=2)

    # a = sqrt(x^2 + y^2) calculates the magnitude of the acceleration vector for each row
    node_acc = np.linalg.norm(node_loc_acc, axis=1)

    result_df = pd.DataFrame({
        'frame_idx': df_track['frame_idx'],
        'track': track,
        'acceleration': node_acc
    })

    return result_df


# ================================ total distance traveled functions ======================================


def calc_total_distance(df, track):
    """
    calc the total distance traveled for a specific track in pixel units

    inputs:
        df: df containing preprocessed tracking data.
        track (str): The track identifier to filter by.

    returns:
        float: total distance traveled in pixel units for the specified track.
    """

    # filtering data for the specified track
    track_data = df[df['track'] == track].sort_values('frame_idx')

    # get our coords based on a central body part (the spine)
    coords = track_data[['spine_adj_x', 'spine_adj_y']].values
    
    # apply the dist formula to consective coords for each x and y col
    # summing across each row

    # this is # sqrt((dx)**2 + (dy)**2)
    dists = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))

    # and we sum all of these dists across all rows
    total_distance = dists.sum()
    return total_distance


def calc_total_distance_traveled(df, track_ids, frame_range=(0, None)):
    """
    calculates the total distance traveled by each track in the df

    inputs: track_ids (a list) and a desired frame range (tuple, all frames by default)
    returns distance_df: a df with the total distance traveled by all tracks
    """
    
    distance_data = []

    # calc the total distance traveled by each track
    for track_id in tqdm(track_ids, desc="Calculating total distance for each track"):
        total_distance = calc_total_distance(df, track_id)
        distance_data.append({'track': track_id, 'total_distance': total_distance})

    # convert the list of dictionaries to a df
    distance_df = pd.DataFrame(distance_data)
    return distance_df



# ================================ wavelet functions ======================================

def perform_cwt_and_compute_energy(velo_df, track, scales, wavelet='mexh'):
    """
    performs continuous wavelet transform (CWT) and computes energy
    """
    velocities = velo_df[track].dropna().values

    # check if the chosen wavelet is real-valued
    if wavelet not in ['mexh', 'morl', 'db4', 'coif1', 'coif2', 'coif3']:
        raise ValueError(f"{wavelet} is not a supported real-valued wavelet. Use a real wavelet like 'mexh', 'db4', 'morl', 'coif'.")

    # perform CWT using wavelets
    coefficients, frequencies = pywt.cwt(velocities, scales, wavelet)
    energy = np.sum(np.abs(coefficients) ** 2, axis=0)

    # no normalization of energy
    # keeping raw energy values for clustering
    return pd.Series(energy, index=velo_df.index[:len(energy)]), coefficients

def aggregate_energy_over_windows(wavelet_energy, window_size):
    """
    aggregates energy values over a specified window size
    """
    wavelet_energy['window'] = (wavelet_energy.index // window_size)
    windowed_energy = wavelet_energy.groupby('window').mean()
    return windowed_energy

def classify_activity_levels(windowed_energy, n_clusters=4):
    """"
    classifies energy values into activity levels using KMeans clustering
    """

    if len(windowed_energy) < n_clusters:
        print(f"not enough samples ({len(windowed_energy)}) for {n_clusters} clusters, reducing clusters to {len(windowed_energy)}.")
        n_clusters = len(windowed_energy)

    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        windowed_energy['cluster'] = kmeans.fit_predict(windowed_energy)

        # sort clusters by mean energy and assign labels
        cluster_centers = kmeans.cluster_centers_
        sorted_clusters = np.argsort(cluster_centers.mean(axis=1))
        labels = ['idle', 'slightly active', 'moderately active', 'very active']
        windowed_energy['activity_level'] = windowed_energy['cluster'].apply(
            lambda x: labels[sorted_clusters.tolist().index(x)]
        )
    else:
        windowed_energy['cluster'] = 0
        windowed_energy['activity_level'] = 'idle'

    return windowed_energy



