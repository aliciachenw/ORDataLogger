
import numpy as np

def createImageCorners(w, h):
    """Create corner coordinates for an image.
    Top-left corner is supposed to be the (0, 0) corner.

    Parameters
    ----------
    w : int
        Image width (in *pixel*)

    h : int
        Image height (in *pixel*)

    pixel2mmX, pixel2mmY : float
        Number of mm for each pixel in US image, for horizontal and vertical axis (in *mm/pixel*)

    Returns
    -------
    np.ndarray
        4 x 4 array of coordinates. Each column is a corner. To (x, y), (z, 1)
        are also added to make them ready to be mulitplied by a roto-translation
        matrix.

    """
    pc = np.array((
            (w,0,0,1),
            (w,h,0,1),
            (0,0,0,1),
            (0,h,0,1),
        )).T
    return pc


def createImageCoords(h, w):
    """Create all pixel coordinates for an image.
    Top-left corner is supposed to be the (0, 0) corner.

    Parameters
    ----------
    w : int
        Image width (in *pixel*)

    h : int
        Image height (in *pixel*)

    Returns
    -------
    np.ndarray
        4 x (w * h) array of coordinates. Each column is a point. To (x, y), (z, 1)
        are also added to make them ready to be mulitplied by a roto-translation
        matrix.

    """
    Np = h * w
    x = np.linspace(0,w-1,w)
    y = np.linspace(0,h-1,h)
    xv, yv = np.meshgrid(x, y)
    xv = np.reshape(xv.ravel(), (1,Np))
    yv = np.reshape(yv.ravel(), (1,Np))
    zv = np.zeros((1,Np))
    b = np.ones((1,Np))
    p = np.concatenate((xv,yv,zv,b), axis=0) # 4 x Np
    return p


def createTopCoords(w):
    """Create all pixel coordinates for scanning plan.
    Top-left corner is supposed to be the (0, 0) corner.

    Parameters
    ----------
    w : int
        Image width (in *pixel*)

    Returns
    -------
    np.ndarray
        4 x (w * h) array of coordinates. Each column is a point. To (x, y), (z, 1)
        are also added to make them ready to be mulitplied by a roto-translation
        matrix.

    """
    Np =  w
    x = np.linspace(0,w-1,w)
    xv = np.reshape(x.ravel(), (1,Np))
    yv = np.zeros((1,Np))
    zv = np.zeros((1,Np))
    b = np.ones((1,Np))
    p = np.concatenate((xv,yv,zv,b), axis=0) # 4 x Np
    return p