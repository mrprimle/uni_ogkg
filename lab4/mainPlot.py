import numpy as np
from delaunay2D import Delaunay2D

if __name__ == '__main__':

    # Generate 'numSeeds' random seeds in a square of size 'radius'
    numSeeds = 24
    radius = 100
    seeds = radius * np.random.random((numSeeds, 2))
    print("seeds:\n", seeds)
    print("BBox Min:", np.amin(seeds, axis=0),
          "Bbox Max: ", np.amax(seeds, axis=0))

    """
    Compute our Delaunay triangulation of seeds.
    """
    # It is recommended to build a frame taylored for our data
    # dt = D.Delaunay2D() # Default frame
    center = np.mean(seeds, axis=0)
    dt = Delaunay2D(center, 50 * radius)
    
    # Insert all seeds one by one
    for s in seeds:
        dt.addPoint(s)

    # Dump number of DT triangles
    print (len(dt.exportTriangles()), "Delaunay triangles")


       
    import matplotlib.pyplot as plt
    import matplotlib.tri
    import matplotlib.collections

    fig, ax = plt.subplots()
    ax.margins(0.1)
    ax.set_aspect('equal')
    plt.axis([-1, radius+1, -1, radius+1])

    # Plot our Delaunay triangulation (plot in blue)
    cx, cy = zip(*seeds)
    dt_tris = dt.exportTriangles()
    ax.triplot(matplotlib.tri.Triangulation(cx, cy, dt_tris), 'bo--')

    # Plot annotated Delaunay vertex (seeds)
    
    # Build Voronoi diagram as a list of coordinates and regions
    vc, vr = dt.exportVoronoiRegions()
    

    # Plot voronoi diagram edges (in red)
    for r in vr:
        polygon = [vc[i] for i in vr[r]]       # Build polygon for each region
        plt.plot(*zip(*polygon), color="red")  # Plot polygon edges in red
    
    

    plt.show()

    # Plot a step-by-step triangulation
    
    # Starts from a new Delaunay2D frame
    
    dt2 = Delaunay2D(center, 50 * radius)    
    for i,s in enumerate(seeds):
        print("Inserting seed", i, s)
        dt2.addPoint(s)
        if i > 1:
            fig, ax = plt.subplots()
            ax.margins(0.1)
            ax.set_aspect('equal')
            plt.axis([-1, radius+1, -1, radius+1])            
            for i, v in enumerate(seeds):
                plt.annotate(i, xy=v)              # Plot all seeds
            for t in dt2.exportTriangles():
                polygon = [seeds[i] for i in t]     # Build polygon for each region
                plt.fill(*zip(*polygon), fill=False, color="b")  # Plot filled polygon

            plt.show()
    
    
    
