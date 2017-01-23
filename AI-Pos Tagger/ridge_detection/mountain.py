#!/usr/bin/python
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2016
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
from scipy.misc import imsave
import scipy.stats
import sys
import copy

# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return filtered_y**2

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( max(y-thickness/2, 0), min(y+thickness/2, image.size[1]-1 ) ):
            image.putpixel((x, t), color)
    return image

def simple(es):
    ret = []
    for col in range(es.shape[1]):
        m = -1
        m_row = -1
        for row in range(es.shape[0]):
            if es[row][col] > m:
                m = es[row][col]
                m_row = row
        ret.append(m_row)

    return ret

tol = 3 # Tolerance of distance between rows in adjacent columns
div = (2*sum(range(tol+1))) # divisor to make a probability distribution (ish)

# returns a probability that favors closer row values over farther values
def getTransition(row1, row2, var = 1):
    dif = abs(row1 - row2)
    if dif > tol:
        return float(0.001 / dif) # this technically makes this not a probability dist anymore, but that's okay
    else:
        return float((tol - dif) / div)
    # return scipy.stats.norm(0, var).pdf(row1 - row2)

# main program
#
(input_filename, output_filename, gt_row, gt_col) = sys.argv[1:]

# load in image
input_image = Image.open(input_filename)

# compute edge strength mask
es = edge_strength(input_image)
imsave('edges.jpg', es)

# takes a distribution as a list of probabilities, and samples from that distribution
def drawFromDist(dist, norm = 0):
    cumm = 0
    x = -1
    p = random.uniform(0, 1)
    m = 1
    if norm == 0:
        m = sum(dist)
    for i in range(len(dist)):
        cumm += float(dist[i] / m)
        if (p < cumm):
            return i

# normalizes a vector. Useful for taking a distribution vector and normalizing (so that it does define a probability distribution)
def rescale(dens):
    m = sum(dens)
    return [float(d/m) for d in dens]

# row probability is the edge strength scaled by the sum over all edge strengths in row
def rProb(row, r):
    m = sum(row)
    return float(r / m)

# gibbs sampling function, takes edge strengths and a "use human input" switch
def gibbs(es, hi = 0):
    T = 500 # number of samples
    burn = 250 # number of initial samples to ignore
    rows = es.shape[0]
    cols = es.shape[1]
    X = zeros(shape=(T+1,cols)) # sample matrix
    tmp = [drawFromDist(es[:, c]) for c in range(es.shape[1])] # initial sample
    X[0] = [tmp[i] for i in range(len(tmp))]

    sums = [sum(es[:,c]) for c in range(cols)]

    for i in range(T):
        X[i+1] = copy.copy(X[i])

        for c in range(es.shape[1] - 1):
            # generate P-density
            p_density = []
            if hi == 1 and c == int(gt_col):
                p_density = [0]*rows
                p_density[int(gt_row)] = 1
            else:
                r_next = X[i+1][c + 1]
                s = sums[c]
                p_density = rescale([ (es[r,c]/s) * getTransition(r, r_next) for r in range(rows) ])
            X[i+1][c] = drawFromDist(p_density, 1)

    return [scipy.stats.mode(X[burn:(T+1), c])[0] for c in range(cols)]

# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.
ridge1 = simple(es)
ridge2 = gibbs(es)
ridge3 = gibbs(es, 1)
# gibbs(es)
# ridge = [ es.shape[0]/2 ] * es.shape[1]

# output answer
edge = draw_edge(input_image, ridge1, (255, 0, 0), 5)
edge = draw_edge(edge, ridge2, (0, 0, 255), 5)
edge = draw_edge(edge, ridge3, (0, 255, 0), 5)
imsave(output_filename, edge)