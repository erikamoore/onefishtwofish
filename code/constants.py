# ==================== numerical constants ========================

# define fps
FPS = 30

# define centers
MID_CX, MID_CY = 524.6622194268969, 568.6876277393765
LS_CX, LS_CY = 549.74621958, 524.28240998


# ==================== other constants ========================
track_ids = ['track_0', 'track_1', 'track_2', 'track_3']  # CHANGE THIS FOR FUTURE DATA (if needed)
parts=['mouth', 'L_eye', 'R_eye', 'spine', 'tail']
edges=[('L_eye', 'mouth'), ('R_eye', 'mouth'), ('L_eye', 'spine'), ('R_eye', 'spine'), ('spine', 'tail')]

# for matplotlib
standard_colors = [
    (0/255, 114/255, 189/255),    # track 0 - Blue
    (217/255, 83/255, 25/255),    # track 1 - Orange
    (237/255, 177/255, 32/255),   # track 2 - Yellow
    (126/255, 47/255, 142/255)    # track 3 - Purple
]

# for when we plot with plotly
px_standard_colors = [
    'rgb(0, 114, 189)',    # Blue
    'rgb(217, 83, 25)',    # Red
    'rgb(237, 177, 32)',   # Yellow
    'rgb(126, 47, 142)'    # Purple
]


general = 'hover over the path to display polar coordinates + travel with the fish'
likely = 'dynamic time warping used to predict frame likely to display coordinated trajectories'
unlikely= 'dynamic time warping used to predict frame unlikely to display coordinated trajectories'