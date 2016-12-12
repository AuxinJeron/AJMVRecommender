from numpy import linalg as la
from scipy import linalg as sla
from sklearn.cluster import KMeans
import igraph as ig
import numpy as np
import math
import plotly.plotly as py
import plotly.graph_objs as go

py.sign_in('panbabybaby', 'JS2Punm2xSGm2MgZEFA3')

def normalized_clutering_ng(S, k):
    print(S)
    rows, cols = S.shape
    D = degree_matrix(S)
    X = la.inv(sla.sqrtm(D))
    W = S
    L = D - W
    L_sym = X * L * X

    # computer the eigenvectors
    eig_val, eig_vec = la.eig(L_sym)
    eig_val = eig_val[:k]
    eig_vec = eig_vec[:k]
    print(eig_val)
    eig_vec = np.transpose(eig_vec)

    # normalize
    U = [0 for i in range(rows)]
    for i in range(rows):
        s = 0
        for j in range(k):
            s += eig_vec[i, j] * eig_vec[i, j]
        U[i] = math.sqrt(s)

    for i in range(rows):
        for j in range(k):
            eig_vec[i, j] = eig_vec[i, j] / U[i]

    kmeans = KMeans(random_state=0).fit(eig_vec)
    return kmeans.labels_

def degree_matrix(S):
    rows, cols = S.shape
    D = np.matrix(np.zeros((rows, cols)))
    for i in range(rows):
        D[i, i] = np.sum(S[i])
        D[i, i] -= S[i, i]
    return D

def create_3D_graph(ratings_mat, users, movies, g):
    nodes, edges = create_graph(ratings_mat, users, movies, g)

    # list of edges
    N = len(nodes)
    L = len(edges)
    Edges = [(edges[k]["source"], edges[k]["target"]) for k in range(L)]
    G = ig.Graph(Edges, directed=False)

    labels = []
    group = []
    for node in nodes:
        labels.append(node["name"])
        group.append(node["group"])
    print(group)
    # get  the node position
    layt = G.layout('kk', dim = 3)
    Xn = [layt[k][0] for k in range(N)]  # x-coordinates of nodes
    Yn = [layt[k][1] for k in range(N)]  # y-coordinates
    Zn = [layt[k][2] for k in range(N)]  # z-coordinates
    Xe = []
    Ye = []
    Ze = []
    for e in Edges:
        Xe += [layt[e[0]][0], layt[e[1]][0], None]  # x-coordinates of edge ends
        Ye += [layt[e[0]][1], layt[e[1]][1], None]
        Ze += [layt[e[0]][2], layt[e[1]][2], None]

    # begin to draw
    trace1 = go.Scatter3d(x=Xe,
                       y=Ye,
                       z=Ze,
                       mode='lines',
                       line=go.Line(color='rgb(125,125,125)', width=1),
                       hoverinfo='none'
                       )
    trace2 = go.Scatter3d(x=Xn,
                       y=Yn,
                       z=Zn,
                       mode='markers',
                       name='actors',
                       marker=go.Marker(symbol='dot',
                                     size=6,
                                     color=group,
                                     colorscale='Viridis',
                                     line=go.Line(color='rgb(50,50,50)', width=0.5)
                                     ),
                       text=labels,
                       hoverinfo='text'
                       )
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )
    layout = go.Layout(
        title="Network of coappearances of characters in Victor Hugo's novel<br> Les Miserables (3D visualization)",
        width=1000,
        height=1000,
        showlegend=False,
        scene=go.Scene(
            xaxis=go.XAxis(axis),
            yaxis=go.YAxis(axis),
            zaxis=go.ZAxis(axis),
        ),
        margin=go.Margin(
            t=100
        ),
        hovermode='closest',
        annotations=go.Annotations([
            go.Annotation(
                showarrow=False,
                text="AJMV Recommender Clustering",
                xref='paper',
                yref='paper',
                x=0,
                y=0.1,
                xanchor='left',
                yanchor='bottom',
                font=go.Font(
                    size=14
                )
            )
        ]), )

    data = go.Data([trace1, trace2])
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='AJMV Recommender Clustering')
    input("Press Enter to quit...")

def create_graph(rating_mat, users, movies, groups):
    group_n = len(groups)
    movie_n = len(movies)
    user_n = len(users)
    N = movie_n + user_n

    # create nodes first user_n nodes are for users, left movie_n nodes are for movies
    nodes = [None for i in range(N)]
    max_group = 0
    for i in range(user_n):
        v = {}
        v["name"] = str(users[i])
        v["group"] = groups[i]
        nodes[i] = v
        max_group = max(max_group, groups[i])
    for i in range(movie_n):
        v = {}
        v["name"] = str(movies[i])
        v["group"] = max_group + 1
        nodes[user_n + i] = v

    #create edges between user and movie
    r,c = rating_mat.shape
    edges = []
    for i in range(r):
        for j in range(c):
            if rating_mat[i, j] <= 3:
                continue
            e = {}
            e["source"] = i
            e["target"] = user_n + j
            e["value"] = math.floor(rating_mat[i, j] + 0.5)
            edges.append(e)
    return nodes, edges

