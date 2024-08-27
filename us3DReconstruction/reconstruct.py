import numpy as np
from us3DReconstruction.image_utils import *
from us3DReconstruction.sanity_utils import *
from us3DReconstruction.voxel_utils import *
import time
import vtk
import SimpleITK as sitk
import cc3d

class Process:
    """Class for performing: voxel-array reconstruction
    """

    def __init__(self):
        """Constructor
        """

        # Data source files
        self.tracking_seq = None  # N x 4 x 4
        self.image_seq = None  # N x height x width
        self.seq_length = None
        # US images parameters
        self.us_width = None
        self.us_height = None

        # Image-to-US probe attitude
        self.T_image2probe = None

        # Calibration results
        self.T_image2world_seq = None

        # Lab-to-conv attitude
        self.T_world2conv = np.eye(4)

        # Frames for voxel array reconstruction
        self.voxFrames = 'auto'

        # Voxel array parameters
        self.xmin = None
        self.ymin = None
        self.zmin = None
        self.xmax = None
        self.ymax = None
        self.zmax = None
        self.xl = None
        self.yl = None
        self.zl = None
        self.xo = None
        self.yo = None
        self.zo = None


        # image scale: pixel/mm
        self.fx = 1.
        self.fy = 1.
        self.fz = 1.

        # US images alignment parameters
        self.wrapper = 'none'
        self.step = 1
        self.alignFrames = None
        self.validKineFrames = None
        self.fillVoxMethod = 'avg'

        # Voxel array data
        self.V = None
        self.contV = None
        self.usedV = None
        self.internalV = None

        # vtkImageData properties
        self.sx = None
        self.sy = None
        self.sz = None

        # Gaps filling parameters
        self.method = 'none'
        self.blocksN = 100
        self.blockDir = 'X'
        self.maxS = 3
        self.distTh = None
        self.minPct = 0.



    def init(self, track_seq, image_seq, T_image2probe=None):
        # set sequence and calibration data

        self.tracking_seq = track_seq 
        self.image_seq = image_seq
        checkDataSequence(self.image_seq, self.tracking_seq)

        self.seq_length = self.image_seq.shape[0]
        self.us_height = self.image_seq.shape[1]
        self.us_width = self.image_seq.shape[2]
        if T_image2probe is None:
            self.T_image2probe = np.eye(4)
        else:
            self.T_image2probe = T_image2probe


    def calculatePoseForUSImages(self):
        # Calculate matrix for pixel to world
        # T_probe_to_world @ T_image_to_probe
        print('Calculating US images roto-translation matrix for all time frames ...')
        self.T_image2world_seq = np.matmul(self.tracking_seq, self.T_image2probe)

    def getImageCornersAs3DPoints(self):
        """Create virtual 3D points for US images corners with respect to the global reference frame.

        Returns
        -------
        dict
            Dictionary where keys are 4 marker names and values are np.ndarray
            N x 3 matrices, representing point coordinates, for N time frames.
            The following are the points created:

            - im_TR: top-right corner
            - im_BR: bottom-right corner
            - im_TL: top-left corner
            - im_BL: bottom-left corner

        """
        # Create virtual points for corners
        pc = createImageCorners(self.us_width, self.us_height)
        pcg = np.matmul(self.T_image2world_seq, pc)[:,0:3,:]    # N x 3 x 4
        points = {}
        points['im_TR'] = pcg[:,:,0]
        points['im_BR'] = pcg[:,:,1]
        points['im_TL'] = pcg[:,:,2]
        points['im_BL'] = pcg[:,:,3]
        return points


    def setValidFramesForVoxelArray(self, voxFrames='auto', voxFramesBounds=None):
        """Set the list of frames (US time line) of the images that can be contained in the voxel array.
        Frames are further filtered out based on the invalid kinematics frames calculated
        by ``calculatePoseForUSProbe()``.

        Parameters
        ----------
        voxFrames : mixed
            List of US time frames.
            If 'auto', all the frames without missing optoelectronic data information will be considered.
            If 'all', all the frames will be considered.
            If list, it must contain the list of frames to be considered.

        voxFramesBounds : mixed
            Bounding frames for the list of frames to be contained in the voxel array.
            If None, all the frames out of ``voxFrames`` will be used.
            If list, it must contain 2 elements specifying lower and upper bround frames for the list in ``voxFrames``.

        """

        # Check input validity
        checkPose(self.T_image2world_seq)
        checkVoxFrames(voxFrames, self.T_image2world_seq.shape[0])
        checkVoxFramesBounds(voxFramesBounds, self.T_image2world_seq.shape[0])

        # Create voxel frames indices
        if voxFrames == 'all':
            voxFrames = range(0, self.T_image2world_seq.shape[0])
        elif voxFrames == 'auto':
            voxFrames = (np.delete(np.arange(self.T_image2world_seq.shape[0]), np.nonzero(np.isnan(self.T_image2world_seq))[0])).tolist()

        # Creae voxel frames bounds if not existing
        if voxFramesBounds is None:
            voxFramesBounds = [0, self.T_image2world_seq.shape[0]-1]

        # Limit voxel frames to bounds
        voxFrames = np.array(voxFrames)
        voxFrames = voxFrames[(voxFrames >= voxFramesBounds[0]) & (voxFrames <= voxFramesBounds[1])]
        self.voxFrames = voxFrames
        # filter out not used data
        ivx = np.array(self.voxFrames)
        self.image_seq = self.image_seq[ivx, :, :]
        self.tracking_seq = self.tracking_seq[ivx, :, :]
        self.T_image2world_seq = self.T_image2world_seq[ivx, :, :]
        self.seq_length = self.image_seq.shape[0]
        self.voxFrames = range(0, self.seq_length)


    def calculateConvPose(self, convR):
        """Calculate roto-translation matrix from global reference frame to *convenient* reference frame.
        Voxel-array dimensions are calculated in this new refence frame. This rotation is important whenever the US scans sihouette is remarkably
        oblique to some axis of the global reference frame. In this case, the voxel-array dimensions (calculated by the smallest parallelepipedon
        wrapping all the realigned scans), calculated in the global refrence frame, would not be optimal, i.e. larger than necessary.

        .. image:: diag_scan_direction.png
           :scale: 30 %

        Parameters
        ----------
        convR : mixed
            Roto-translation matrix.
            If str, it specifies the method for automatically calculate the matrix.
            If 'auto_PCA', PCA is performed on all US image corners. The x, y and z of the new convenient reference frame are represented by the eigenvectors out of the PCA.
            If 'first_last_frames_centroid', the convenent reference frame is expressed as:

            - x from first image centroid to last image centroid
            - z orthogonal to x and the axis and the vector joining the top-left corner to the top-right corner of the first image
            - y orthogonal to z and x

            If np.ndarray, it must be manually specified as a 4 x 4 affine matrix.

        """

        # Check input validity
        checkPose(self.T_image2world_seq)
        checkVoxFrames(self.voxFrames, self.T_image2world_seq.shape[0])
        checkGl2ConvPose(convR)

        if not isinstance(convR, str):
            self.T_world2conv = convR
        else:
            # Calculating best pose automatically, if necessary
            pc = createImageCorners(self.us_width, self.us_height)
            self.T_world2conv = convR
            if self.T_world2conv == 'auto_PCA':
                # Perform PCA on image corners
                print('Performing PCA on images corners...')
                pcg = np.matmul(self.T_image2world_seq, pc)[:,0:3,:]    # N x 3 x 4
                pcg = np.reshape(pcg.transpose(1,2,0), (3, 4 * self.seq_length), order='F')
                U, s = pca(pcg)
                # Build convenience affine matrix
                self.T_world2conv = np.vstack((np.hstack((U, np.zeros((3,1)))),[0,0,0,1])).T
                print('PCA perfomed')
            elif self.T_world2conv == 'first_last_frames_centroid':
                # Search connection from first image centroid to last image centroid (X)
                print('Performing convenient reference frame calculation based on first and last image centroids...')
                pcg = np.matmul(self.T_image2world_seq, pc)[:,0:3,:]   # N x 3 x 4
                C0 = np.mean(pcg, axis=1)  # 3
                C1 = np.mean(pcg, axis=1)  # 3
                X = C1 - C0
                # Define Y and Z axis
                corners0 = pcg[0,:,:] # 3 x 4
                Ytemp = corners0[:,0] - corners0[:,2]   # from top-left corner to top-right corner
                Z = np.cross(X, Ytemp)
                Y = np.cross(Z, X)
                # Normalize axis length
                X = X / np.linalg.norm(X)
                Y = Y / np.linalg.norm(Y)
                Z = Z / np.linalg.norm(Z)
                # Create rotation matrix
                M = np.array([X, Y, Z]).T
                # Build convenience affine matrix
                self.T_world2conv = np.vstack((np.hstack((M, np.zeros((3,1)))), [0,0,0,1])).T
                print('Convenient reference frame calculated')


    def getVoxelArrayPose(self):
        """Return roto-translation matrix from voxel array reference frame to global reference frame.

        Returns
        -------
        np.ndarray
            4 x 4 rototranslation matrix.

        """

        # Define roto-translation from convenient to global reference frame
        Tconv = np.linalg.inv(self.T_world2conv)

        # Define roto-translation from voxel-array to convenient reference frame
        convTva = np.eye(4)
        convTva[0:3,3] = [self.xmin,self.ymin,self.zmin]

        # Define roto-translation from voxel-array to global reference frame
        Tva = np.matmul(Tconv, convTva)

        return Tva



    def setScaleFactors(self, fxyz, voxFramesBounds=None):
        """Set or calculate scale factors that multiply real voxel-array dimensions.

        Parameters
        ----------
        fxyz : mixed
            Scale factors.
            If list, it must contain 3 elements being the scale factors
            If 'auto_bounded_parallel_scans', the following should hold:

            - the US probe motion is supposed to be performed mainly along one axis (X);
            - corners of the US images during acquisition are supposed to not deviate too much from a straight line (along X);
            - motion velocity is supposed to be constant;
            - pixel/mm for US images are very similar for width and height.

            Scale factors are calculated as follows:

            - fx: ceil(abs((voxFramesBounds[1] - voxFramesBounds[0]) / (C1 - C0)));
            - fy, fz: ceil(1 / pixel2mmX).

            where:

            - C0 and C1 are the X coordinates (in *mm*) of the US image centers at frames ``voxFramesBounds[0]`` and ``voxFramesBounds[1]``;
            - pixel2mmX is the conversion factor (in *mm/pixel*) for width in the US images.

            See chapter :ref:`when-mem-error` for the use of these scale factors.

        voxFramesBounds : mixed
            Bounding frames for the list of frames to be contained in the voxel array.
            If None, first and last time frames out of ``setValidFramesForVoxelArray()`` will be used.
            If list, it must contain 2 elements specifying lower and upper bround frames.

        """

        # Check input validity
        checkFxyz(fxyz)
        # checkPose(self.T_image2world_seq)
        # checkVoxFrames(self.voxFrames, self.T_image2world_seq.shape[0])
        # checkVoxFramesBounds(voxFramesBounds, self.T_image2world_seq.shape[0])
        # checkGl2ConvPose(self.T_world2conv)

        # Creae voxel frames bounds if not existing
        if voxFramesBounds is None:
            voxFramesBounds = [self.voxFrames[0], self.voxFrames[-1]]

        fx, fy, fz = fxyz[0], fxyz[1], fxyz[2]
        self.fx, self.fy, self.fz = fx, fy, fz
        self.sx = 1.0 / fx
        self.sy = 1.0 / fy
        self.sz = 1.0 / fz
        print('Scale factors fx, fy, fz set to: %d, %d, %d' % (self.fx, self.fy, self.fz))



    def calculateVoxelArrayDimensions(self):
        """Calculate dimensions for voxel array. The convenient reference frame
        (see ``calculateConvPose()``) is translated to a *voxel array* reference
        frame, optimally containing the US images is the first quadrant.
        """

        # Check input validity
        checkFxyz([self.fx, self.fy, self.fz])
        checkPose(self.T_image2world_seq)
        checkGl2ConvPose(self.T_world2conv)
        checkVoxFrames(self.voxFrames, self.T_image2world_seq.shape[0])

        # Calculate coordinates for all points in convevient reference frame
        pc = createImageCorners(self.us_width, self.us_height)
        pcg = np.matmul(np.matmul(self.T_world2conv, self.T_image2world_seq), pc)  # N x 4 x 4 (#frames x #coords+1 x #points)

        # Calculate volume dimensions
        print('Calculating voxel array dimension ...')
        xmin, xmax = np.min(pcg[:,0,:]), np.max(pcg[:,0,:])
        ymin, ymax = np.min(pcg[:,1,:]), np.max(pcg[:,1,:])
        zmin, zmax = np.min(pcg[:,2,:]), np.max(pcg[:,2,:])
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.zmin, self.zmax = zmin, zmax
        # Calculate voxel array size
        self.xl = (np.round(self.fx * xmax) - np.round(self.fx * xmin)) + 1
        self.yl = (np.round(self.fy * ymax) - np.round(self.fy * ymin)) + 1
        self.zl = (np.round(self.fz * zmax) - np.round(self.fz * zmin)) + 1
        self.xo = np.round(self.fx * xmin)
        self.yo = np.round(self.fy * ymin)
        self.zo = np.round(self.fz * zmin)
        print('Voxel array dimension: {0} x {1} x {2}'.format(int(self.xl), int(self.yl), int(self.zl)))


    def initVoxelArray(self):
        """Initialize voxel array. It instantiate data for the voxel array grey values.
        """

        # Create voxel array for grey values
        self.V = VoxelArray3DFrom2DImages(dataType=np.uint8, dims=(self.xl,self.yl,self.zl), scales=(self.fx,self.fy,self.fz))


    def setUSImagesAlignmentParameters(self, **kwargs):
        """Set parameters for US scans alignement in global reference frame.
        See chapter :ref:`when-mem-error` for tips about setting these parameters.

        Parameters
        ----------
        wrapper : str
            Type of wrapper to create scanning silhouette.
            If 'parallelepipedon', the smallest wrapping paralellepipedon (with
            dimensions aligned with the global reference frame) is created between
            two US scans.
            If 'convex_hull', the convex hull is created between two US scans.
            This one is more accurate than 'parallelepipedon', but it takes more
            time to be created.
            If 'none' (default), no wrapper is created.

            .. image:: parall_vs_convexhull.png
               :scale: 50 %

        step : int
            Interval (in number of US frames) between two US scans
            used to create the wrapper. Default to 1.

        alignFrames : list
            List of frames (US time line) on which to perform US images alignment.

        fillVoxMethod : str
            Method for filling each voxel.
            If 'avg', an average between the current voxel value and the new value
            is performed.
            If 'last', the new voxel value will replace the current one.
            If 'max', the highest voxel value will replace the current one.

        """

        # Check wrapper
        if 'wrapper' in kwargs:
            wrapper = kwargs['wrapper']
            checkWrapper(wrapper)
            self.wrapper = wrapper

        # Check step
        if 'step' in kwargs:
            step = kwargs['step']
            checkStep(step)
            self.step = step

        # Check frameWin
        if 'alignFrames' in kwargs:
            alignFrames = kwargs['alignFrames']
            checkPose(self.T_image2world_seq)
            checkAlignFrames(alignFrames, self.T_image2probe.shape[0])
            self.alignFrames = alignFrames

        # Check fillVoxMethod
        if 'fillVoxMethod' in kwargs:
            fillVoxMethod = kwargs['fillVoxMethod']
            checkFillVoxMethod(fillVoxMethod)
            self.fillVoxMethod = fillVoxMethod



    def alignUSImages(self, compoundWhenOverlap=False, Nr=10, Nrest=30, pctIntTh=50., resetAdjRotoTranslAfterCompound=True, alwaysAcceptCompound=False):
        """Align US images in the global reference frame.
        This task can take some time, and computation time is proportional
        to the *total* number of US images to align.

        """

        # Check input validity
        checkPose(self.T_image2world_seq)
        checkGl2ConvPose(self.T_world2conv)
        checkFxyz([self.fx, self.fy, self.fz])
        # xl, xo
        checkV(self.V)
        checkWrapper(self.wrapper)
        checkStep(self.step)
        checkFillVoxMethod(self.fillVoxMethod)

        # Create if necessary and check alignFrames
        if self.alignFrames is None:
            self.alignFrames = range(0, self.T_image2world_seq.shape[0])
        checkAlignFrames(self.alignFrames, self.T_image2world_seq.shape[0])

        # Create pixel coordinates (in mm) in image reference frame
        print('Creating pixel 3D coordinates in image reference frame ...')

        # Calculate image corners coordinates
        pc = createImageCorners(self.us_width, self.us_height)
        p = createImageCoords(self.us_height, self.us_width)

        # Calculate position for all the pixels, for all the time instant
        t = time.time()
        state = 'write_main_VA'

        Aa, ta, ca = np.eye(3), np.zeros((3,1)), np.zeros((3,1))

        iStart = None
        current_frame_idx = 0
        # if compoundWhenOverlap:
        #     secV = VoxelArray3DFrom2DImages(dataType=np.uint8, scales=(self.fx,self.fy,self.fz))
        #     pctIntVect = np.array([])
        #     iIntVect = np.array([])
        #     xMinPrev, yMinPrev, zMinPrev = None, None, None
        #     xMaxPrev, yMaxPrev, zMaxPrev = None, None, None
        #     iTemp = None

        while current_frame_idx < self.seq_length:
            
            # # Create gray values
            # I = self.image_seq[current_frame_idx, :, :].flatten()
            # print('Inserting oriented slice for instant {0}/{1} ...'.format(current_frame_idx, self.seq_length-1))

            # pcg = np.matmul(np.matmul(self.T_world2conv,self.T_image2world_seq[current_frame_idx,:,:]), pc) # mm
            # xc = (np.round(pcg[0,:] * self.fx) - self.xo).squeeze() # 1 x 4
            # yc = (np.round(pcg[1,:] * self.fy) - self.yo).squeeze()
            # zc = (np.round(pcg[2,:] * self.fz) - self.zo).squeeze()

            # xyzc = np.matmul(Aa, np.array((xc,yc,zc,)) - ca) + ta + ca
            # xc, yc, zc = xyzc[0,:], xyzc[1,:], xyzc[2,:]
            # xc = xc.squeeze().round()
            # yc = yc.squeeze().round()
            # zc = zc.squeeze().round()
            # # Calculate frames position in space
            # pg = np.matmul(np.matmul(self.T_world2conv,self.T_image2world_seq[current_frame_idx,:,:]), p) # mm
            # x = (np.round(pg[0,:] * self.fx) - self.xo).squeeze() # 1 x Np
            # y = (np.round(pg[1,:] * self.fy) - self.yo).squeeze()
            # z = (np.round(pg[2,:] * self.fz) - self.zo).squeeze()

            # xyz = np.matmul(Aa, np.array((x,y,z,)) - ca) + ta + ca
            # x, y, z = xyz[0,:], xyz[1,:], xyz[2,:]
            # x = x.squeeze().round()
            # y = y.squeeze().round()
            # z = z.squeeze().round()
            # # Transform coordinates to indices
            # idxV = xyz2idx(x, y, z, self.xl, self.yl, self.zl)
            # # Check intersection persentage between current frame and previous silhouette
            # pctInt = 100. * np.sum(self.V.getSilhouetteVoxelArray().getDataByIdx(idxV)) / idxV.shape[0]
            # print('Intersection percentage (between main voxel-array and current image): %s' % pctInt)

            # # Manage state # TODO: don't know what this is doing, use False to skip this for now??
            # if compoundWhenOverlap:
            #     skipCurrent = False
            #     if state == 'write_main_VA':
            #         if i < self.seq_length - Nr:
            #             if pctInt >= pctIntTh and current_frame_idx >= 1:
            #                 # Start writing to secondary VA
            #                 state = 'write_secondary_VA'
            #                 iTemp = i
            #                 print('Started writing to secondary voxel-array ...')
            #     if state == 'write_secondary_VA':
            #         if (i - iTemp) < Nr:
            #             # Add some statistics about current frame
            #             pctIntVect = np.append(pctIntVect, pctInt)
            #             iIntVect = np.append(iIntVect, i)
            #         else:
            #             # Nr frames assessed
            #             # can change criteria here, and can use pctIntVect, iIntVect
            #             print(iIntVect)

            #             #criteria = True
            #             criteria = np.sum(pctIntVect > pctIntTh) > 0.5 * Nr
            #             if criteria:
            #                 xa2, ya2, za2 = xMinPrev, yMinPrev, zMinPrev
            #                 xb2, yb2, zb2 = xMaxPrev, yMaxPrev, zMaxPrev
            #                 # Slice main silhouette voxel-array using borders of second one
            #                 subInternalV1 = self.V.getSilhouetteVoxelArray().getSubVoxelArray(xa2, xb2, ya2, yb2, za2, zb2)
            #                 # Find smallest parallelepipedon edges containing sliced main silhouette
            #                 xa1, xb1, ya1, yb1, za1, zb1 = subInternalV1.getCoordsSmallestWrappingParallelepipedon()
            #                 xa1 += xa2; xb1 += xa2
            #                 ya1 += ya2; yb1 += ya2
            #                 za1 += za2; zb1 += za2
            #                 # Restrict region of interest
            #                 rx, ry, rz = 1, 1, 1
            #                 xra1 = 0.5 * (xa1 + xb1) - rx * 0.5 * (xb1 - xa1)
            #                 xrb1 = 0.5 * (xa1 + xb1) + rx * 0.5 * (xb1 - xa1)
            #                 yra1 = 0.5 * (ya1 + yb1) - ry * 0.5 * (yb1 - ya1)
            #                 yrb1 = 0.5 * (ya1 + yb1) + ry * 0.5 * (yb1 - ya1)
            #                 zra1 = 0.5 * (za1 + zb1) - rz * 0.5 * (zb1 - za1)
            #                 zrb1 = 0.5 * (za1 + zb1) + rz * 0.5 * (zb1 - za1)
            #                 # Slice main voxel-array using final borders
            #                 subV1 = self.V.getSubVoxelArray(xra1, xrb1, yra1, yrb1, zra1, zrb1)
            #                 # Slice secondary voxel-array using final borders
            #                 subV2 = secV.getSubVoxelArray(xra1-xa2, xrb1-xa2, yra1-ya2, yrb1-ya2, zra1-za2, zrb1-za2)
            #                 # Fill gaps for both sub-volumes
            #                 subV1.fillGaps(method=self.method, blocksN=1, blockDir=self.blockDir, distTh=self.distTh, maxS=self.maxS, minPct=self.minPct)
            #                 subV2.fillGaps(method=self.method, blocksN=1, blockDir=self.blockDir, distTh=self.distTh, maxS=self.maxS, minPct=self.minPct)
            #                 # Apply 3D registration
            #                 V1 = subV1.getNumpyArray3D()
            #                 V2 = subV2.getNumpyArray3D()
            #                 S1 = 255*subV1.getSilhouetteVoxelArray().getNumpyArray3D().astype(np.uint8)
            #                 S2 = 255*subV2.getSilhouetteVoxelArray().getNumpyArray3D().astype(np.uint8)
            #                 print('Performing 3D registration ...')
            #                 _A, _t, _c = compound3D(V1, V2, S1, S2, (xra1, yra1, zra1))
            #                 if alwaysAcceptCompound:
            #                     Aa, ta, ca = _A, _t, _c
            #                 else:
            #                     choice = input('Keep it? (y/n, n=use previous)? ')
            #                     if choice == 'y':
            #                         Aa, ta, ca = _A, _t, _c
            #                 print('3D registration performed')
            #             else:
            #                 # Do not apply registration
            #                 print('3D registration will not be applied')
            #             # Reset to default state which also ignores intersection (otherwise infinite loop)
            #             state = 'write_main_VA_ignore_int'
            #             j = i # temporarily save i into j
            #             i = iTemp
            #             iTemp = j
            #             print('Returning to frame where writing to secondary frame started ...')
            #             skipCurrent = True
            #     elif state == 'write_main_VA_ignore_int':
            #         print(i, iTemp)
            #         if i == iTemp:
            #             # Reset to default state
            #             state = 'write_main_VA'
            #             iTemp = None
            #             pctIntVect = np.array([])
            #             iIntVect = np.array([])
            #             secV = VoxelArray3DFrom2DImages(dataType=np.uint8, scales=(self.fx,self.fy,self.fz))
            #             xMinPrev, yMinPrev, zMinPrev = None, None, None
            #             xMaxPrev, yMaxPrev, zMaxPrev = None, None, None
            #             if resetAdjRotoTranslAfterCompound:
            #                 Aa, ta, ca = np.eye(3), np.zeros((3,1)), np.zeros((3,1))
            #             print('Started writing to main voxel-array ...')
            #             V1 = self.V.getNumpyArray3D()
            #             import SimpleITK as sitk
            #             sitk.Show(sitk.GetImageFromArray(V1))
            #     if skipCurrent:  # go to next iteration
            #         print('Skipping current frame ...')
            #         continue


            # Create gray values
            I = self.image_seq[current_frame_idx, :, :].flatten()
            print('Inserting oriented slice for instant {0}/{1} ...'.format(current_frame_idx, self.seq_length-1))

            pcg = np.matmul(np.matmul(self.T_world2conv, self.T_image2world_seq[current_frame_idx,:,:]), pc) # mm
            xc = (np.round(pcg[0,:] * self.fx) - self.xo).squeeze() # 1 x 4
            yc = (np.round(pcg[1,:] * self.fy) - self.yo).squeeze()
            zc = (np.round(pcg[2,:] * self.fz) - self.zo).squeeze()

            # Calculate frames position in space
            pg = np.matmul(np.matmul(self.T_world2conv, self.T_image2world_seq[current_frame_idx,:,:]), p) # mm
            x = (np.round(pg[0,:] * self.fx) - self.xo).squeeze() # 1 x Np
            y = (np.round(pg[1,:] * self.fy) - self.yo).squeeze()
            z = (np.round(pg[2,:] * self.fz) - self.zo).squeeze()
            # Transform coordinates to indices

            idxV = xyz2idx(x, y, z, self.xl, self.yl, self.zl)

            # Check intersection persentage between current frame and previous silhouette
            pctInt = 100. * np.sum(self.V.getSilhouetteVoxelArray().getDataByIdx(idxV)) / idxV.shape[0]
            print('Intersection percentage (between main voxel-array and current image): %s' % pctInt)

            # Print current state
            print('-- STATE: %s' % state)
            # Select voxel array upon state
            # Calculate image corners
            corners = (xc, yc, zc)
            # Select main voxel-array
            V = self.V
            
            ##### coumpounding function is here

            # Write to selected voxel-array
            V.writeImageByIdx(idxV, I, self.fillVoxMethod)
            # Create wrapper for selected voxel-array
            if current_frame_idx == iStart:
                cornersPrev = corners
            else:
                cornersPrev = None
            updateWrapper = True
            if current_frame_idx < self.seq_length - 1:
                if current_frame_idx % self.step:
                    updateWrapper = False
            if updateWrapper:
                V.updateWrapper(self.wrapper, corners, cornersPrev=cornersPrev)
            # Update frame index
            current_frame_idx += 1

        elapsed = time.time() - t
        print('Elapsed time: {0} s'.format(elapsed))
        V = self.V.getNumpyArray1D()
        usedV = self.V.getCounterVoxelArray().getNumpyArray1D() > 0
        internalV = self.V.getSilhouetteVoxelArray().getNumpyArray1D()
        idxEmptyN = np.sum(~usedV)
        pctEmpty = 100.0 * idxEmptyN / V.size
        print('Pct of empty voxels: ({0}% total)'.format(pctEmpty))
        pctInternal = 100.0 * np.sum(internalV) / V.size
        print('Estimate of pct of internal voxels: ({0}% total)'.format(pctInternal))
        if np.sum(internalV) > 0:
            pctInternalEmpty = 100.0 * np.sum(internalV & ~usedV) / np.sum(internalV)
        else:
            pctInternalEmpty = 0.
        print('Estimate of pct of internal empty voxels: ({0}% internal)'.format(pctInternalEmpty))




    def getScanSurface(self):
        """Align US images in the global reference frame.
        This task can take some time, and computation time is proportional
        to the *total* number of US images to align.

        """

        # Check input validity
        checkPose(self.T_image2world_seq)
        checkGl2ConvPose(self.T_world2conv)
        checkFxyz([self.fx, self.fy, self.fz])
        # xl, xo
        checkV(self.V)
        checkWrapper(self.wrapper)
        checkStep(self.step)
        checkFillVoxMethod(self.fillVoxMethod)

        # Create if necessary and check alignFrames
        if self.alignFrames is None:
            self.alignFrames = range(0, self.T_image2world_seq.shape[0])
        checkAlignFrames(self.alignFrames, self.T_image2world_seq.shape[0])

        # Create pixel coordinates (in mm) in image reference frame
        print('Creating pixel 3D coordinates in image reference frame ...')

        # Calculate image corners coordinates
        pc = createImageCorners(self.us_width, self.us_height)
        p = createImageCoords(self.us_height, self.us_width)

        # Calculate position for all the pixels, for all the time instant
        t = time.time()
        current_frame_idx = 0

        surface_points = []

        state = 'write_main_VA'

        Aa, ta, ca = np.eye(3), np.zeros((3,1)), np.zeros((3,1))

        iStart = None
        current_frame_idx = 0

        while current_frame_idx < self.seq_length:

            # Create gray values
            I = self.image_seq[current_frame_idx, :, :]
            I = np.zeros_like(I)
            I[0,:] = 1
            I = I.flatten()
            print('Inserting oriented slice for instant {0}/{1} ...'.format(current_frame_idx, self.seq_length-1))

            pcg = np.matmul(np.matmul(self.T_world2conv, self.T_image2world_seq[current_frame_idx,:,:]), pc) # mm
            xc = (np.round(pcg[0,:] * self.fx) - self.xo).squeeze() # 1 x 4
            yc = (np.round(pcg[1,:] * self.fy) - self.yo).squeeze()
            zc = (np.round(pcg[2,:] * self.fz) - self.zo).squeeze()

            # Calculate frames position in space
            pg = np.matmul(np.matmul(self.T_world2conv, self.T_image2world_seq[current_frame_idx,:,:]), p) # mm
            x = (np.round(pg[0,:] * self.fx) - self.xo).squeeze() # 1 x Np
            y = (np.round(pg[1,:] * self.fy) - self.yo).squeeze()
            z = (np.round(pg[2,:] * self.fz) - self.zo).squeeze()
            # Transform coordinates to indices

            idxV = xyz2idx(x, y, z, self.xl, self.yl, self.zl)

            # Check intersection persentage between current frame and previous silhouette
            pctInt = 100. * np.sum(self.V.getSilhouetteVoxelArray().getDataByIdx(idxV)) / idxV.shape[0]
            print('Intersection percentage (between main voxel-array and current image): %s' % pctInt)

            # Print current state
            # Select voxel array upon state
            # Calculate image corners
            corners = (xc, yc, zc)
            # Select main voxel-array
            V = self.V
            
            ##### coumpounding function is here

            # Write to selected voxel-array
            V.writeImageByIdx(idxV, I, self.fillVoxMethod)
            # Create wrapper for selected voxel-array
            if current_frame_idx == iStart:
                cornersPrev = corners
            else:
                cornersPrev = None
            updateWrapper = True
            if current_frame_idx < self.seq_length - 1:
                if current_frame_idx % self.step:
                    updateWrapper = False
            if updateWrapper:
                V.updateWrapper(self.wrapper, corners, cornersPrev=cornersPrev)
            # Update frame index
            current_frame_idx += 1

        elapsed = time.time() - t
        print('Elapsed time: {0} s'.format(elapsed))
        V = self.V.getNumpyArray1D()
        usedV = self.V.getCounterVoxelArray().getNumpyArray1D() > 0
        internalV = self.V.getSilhouetteVoxelArray().getNumpyArray1D()
        idxEmptyN = np.sum(~usedV)
        pctEmpty = 100.0 * idxEmptyN / V.size
        print('Pct of empty voxels: ({0}% total)'.format(pctEmpty))
        pctInternal = 100.0 * np.sum(internalV) / V.size
        print('Estimate of pct of internal voxels: ({0}% total)'.format(pctInternal))
        if np.sum(internalV) > 0:
            pctInternalEmpty = 100.0 * np.sum(internalV & ~usedV) / np.sum(internalV)
        else:
            pctInternalEmpty = 0.
        print('Estimate of pct of internal empty voxels: ({0}% internal)'.format(pctInternalEmpty))



    def setGapFillingParameters(self, **kwargs):
        """Set parameters for gap filling.

        Parameters
        ----------
        method : str
            Method for filling gaps.
            If 'VNN' (Voxel Nearest Neighbour, default), the nearest voxel to the gap is
            used to fill the gap. Arguments ``maxS`` and ``minPct`` will be ignored.
            If ``distTh` is set, voxels with a distance greater than this threshold will
            be ignored when filling gaps.
            If 'AVG_CUBE', this procedure is applied:
                1. create a cube with side 3 voxels, centered around the gap
                2. search for a minimum ``minPct`` percentage of non-gaps inside the cube (100% = number of voxels in the cube)
                3. if that percentage is found, a non-gap voxels average (wighted by the Euclidean distances) is performed into the cube
                4. if that percentage is not found, the cube size in incremented by 2 voxels
                5. if cube size is lesser than maxS, start again from point 2. Otherwise, stop and don't fill the gap.
            This method is much slower than 'VNN', but allows to limit the search area.

        maxS : int
            See ``method``. This number must be an odd number. Default to 1.

        minPct : float
            See ``method``. This value must be between 0 and 1. Default to 0.

        blocksN : int
            Positive number (greater or equal than 1) indicating the number of
            subvoxel-arrays into which to decompose the gap-filling problem. This can be tuned to
            modify computation time and memory usage. Default to 100.

        blockDir : str
            String defining the direction for blocks motion.
            It can be 'X', 'Y', 'Z'.

        distTh : int
            See ``method``. This must be greater or equal than 1.

        Notes
        -----
        *Only* the gaps internal to the wrapper created by ``alighImages()`` will beconsidered.
        If a gap is not filled, its value will be considered the same as a *completely black* voxel.
        See chapter :ref:`when-mem-error` for tips about setting these parameters.

        """

        # Check method
        if 'method' in kwargs:
            method = kwargs['method']
            checkMethod(method)
            self.method = method

        # Check blocksN
        if 'blocksN' in kwargs:
            blocksN = kwargs['blocksN']
            checkBlocksN(blocksN)
            self.blocksN = blocksN

        # Check blockDir
        if 'blockDir' in kwargs:
            blockDir = kwargs['blockDir']
            checkBlockDir(blockDir)
            self.blockDir = blockDir

        # Check maxS
        if 'maxS' in kwargs:
            maxS = kwargs['maxS']
            checkMaxS(maxS)
            self.maxS = maxS

        # Check distTh
        if 'distTh' in kwargs:
            distTh = kwargs['distTh']
            checkDistTh(distTh)
            self.distTh = distTh


        # Check minPct
        if 'minPct' in kwargs:
            minPct = kwargs['minPct']
            checkMinPct(minPct)
            self.minPct = minPct


    def fillGaps(self):
        """Run the gap-filling procedure.
        This task can take some time.

        """

        # Check input validity
        checkMethod(self.method)
        checkBlocksN(self.blocksN)
        checkMaxS(self.maxS)
        checkBlockDir(self.blockDir)
        if self.method == 'VNN':
            checkDistTh(self.distTh)
        if self.method == 'AVG_CUBE':
            checkMinPct(self.minPct)
        checkV(self.V)
        pctInternalEmpty = self.V.fillGaps(method=self.method, blocksN=self.blocksN, blockDir=self.blockDir, distTh=self.distTh, maxS=self.maxS, minPct=self.minPct)
        return pctInternalEmpty



    def getVoxelPhysicalSize(self):
        """Get physical size for a single voxel.

        Returns
        -------
        list
            3-elem list with voxel dimensions (in *mm*) for each direction.

        """

        # Check fxyz
        checkFxyz([self.fx, self.fy, self.fz])

        # Calculate physical dimensions (in mm)
        vx = 1. / self.fx
        vy = 1. / self.fy
        vz = 1. / self.fz

        return vx, vy, vz
    

    def setImageDataProperties(self, sxyz):
        self.sx = sxyz[0]
        self.sy = sxyz[1]
        self.sz = sxyz[2]


    def writeData(self, path, data_type=None):
        origin = (self.xo*self.sx, self.yo*self.sy, self.zo*self.sz) # should be equal to self.xmin, self.ymin, self.zmin
        
        # if data_type == 'label':
        #     for label in [1,2,3]:
        #         mask = self.V.V == label
        #         if np.sum(mask) > 0:
        #             # label_region = cc3d.connected_components(mask, connectivity=26)
        #             # label_region = mask.astype(np.uint8)
        #             # label_region = cc3d.dust(label_region, threshold=10, connectivity=26, in_place=False)
        #             # label_region = label_region > 0
        #             # self.V.V[label_region] = label
        #             labels_out, N = cc3d.largest_k(mask.astype(np.uint8), k=10, connectivity=26, delta=0,return_N=True) ## memory issue?
        #             noise = np.logical_and(labels_out == 0, mask)
        #             self.V.V[noise] = 0
        #             # labels_in *= (labels_out > 0) # to get original labels


        sitk_volume = nparray2vtkImageData(self.V.V, (self.xl, self.yl, self.zl), (self.sx, self.sy, self.sz), origin, vtk.VTK_UNSIGNED_CHAR)
        # if data_type == 'label':
        #     # do some morphological operations
        #     carotid = sitk_volume == 1
        #     filter = sitk.BinaryDilateImageFilter()
        #     # filter = sitk.BinaryMorphologicalOpeningImageFilter()
        #     # filter.SetKernelType(sitk.BinaryMorphologicalOpeningImageFilter.Ball)
        #     filter.SetKernelRadius(1)
        #     carotid_opened = filter.Execute(carotid)
        #     carotid_mask = carotid_opened > 0

        #     vein = sitk_volume == 3
        #     vein_opened = filter.Execute(vein)
        #     vein_mask = vein_opened > 0

        #     larynx = sitk_volume == 2
        #     # filter = sitk.BinaryMorphologicalClosingImageFilter()
        #     # filter.SetKernelType(sitk.BinaryMorphologicalClosingImageFilter.Ball)
        #     # filter.SetKernelRadius(1)
        #     larynx_closed = filter.Execute(larynx)
        #     larynx_mask = larynx_closed > 0


        # #     sitk_volume[larynx_mask] = 2
        # #     sitk_volume[carotid_mask] = 1
        # #     sitk_volume[vein_mask] = 3
        # if data_type == 'label':
        #     sitk_volume = sitk.Median(sitk_volume, (3,3,3))
        sitk.WriteImage(sitk_volume, path)
