import sys
sys.path.append("..")

from mifcraft import MIF

# This is code taken from an actual experiment.
# We're going to wrap gold pads around a permalloy wire at fixed intervals,
# generating an Oe field and reducing the current through permalloy. We've 
# shown theoretically that we can freely reduce the field or current drop 
# independently by changing the gold pad geometry, so we want to do
# an experiment where we vary those freely - and cover both senses of the Oe
# field.

# Ratios of field and current drop at their maxima.
H_TO_U0_MAX = 0.828 # Vary freely down to 0
U_TO_U0_MAX = 0.22 # Vary freely down to 0

# Conversion factors
MULTIPLIER = 79.577472 # G to A/m

# Uniform DC current (m/s)
u0 = 160

# Geometric parameters of the system
L = 10000e-9
W = 140e-9
H = 10e-9

# How far from the end of the bar to put the first gold pad
firstPad = 3070e-9
# How long each gold pad is (and how long the space between each gold pad is)
padWidth = 100e-9

# Let's plan:
#    u0 = 160 m/s
#    Vary the following:
#    u/u0 from 0.02 to 0.22 by 0.01 (20 steps)
#    H/u0 from 0.0276 to 0.828 by 0.0276 (30 steps)

# We also specifically need the sim with no H!
# Here's our support function

def quickMIF(Hmult, umult, signMult, signName):
    with MIF(filename='mifs/ym_H'+str(Hmult)+"_u"+str(umult)+".mif",
             basename='../data/'+signName+'/H_'+str(Hmult)+"/u_"+str(umult)+"/out") as M:

        # Define the world and the fixed-spin ends of the bar.
        world  = M.BoxAtlas( xrange = (0, L),        
                             yrange = (0, W), 
                             zrange = (0, H) )
        left   = M.BoxAtlas( xrange = (0, 50e-9),    
                             yrange = (0, W), 
                             zrange = (0, H) )
        right  = M.BoxAtlas( xrange = (L-50e-9, L),  
                             yrange = (0, W), 
                             zrange = (0, H) )
        leadin = M.BoxAtlas( xrange = (0, firstPad), 
                             yrange = (0, W), 
                             zrange = (0, H) )
        edges  = M.MultiAtlas( atlases = [left, right] )

        goldSpots = []
        bareSpots = []

        location = firstPad
        finalLocation = 9500e-9

        # Append alternating gold and bare-Py regions until doing so would run
        # off the end of the bar.
        while True:
            if location + padWidth > finalLocation:
                break
            # Otherwise, we can make a new Au pad
            goldSpots.append( M.BoxAtlas( xrange = (location, 
                                                    location + padWidth),
                                          yrange = (0, W),
                                          zrange = (0, H)))

            location += padWidth
            # Now try a bare Py spot
            bareSpots.append( M.BoxAtlas( xrange = (location, 
                                                    location + padWidth),
                                          yrange = (0, W),
                                          zrange = (0, H)))
            location += padWidth

        # We can't fit another full pair of regions, so pad Py to the end 
        # of the bar
        leadout = M.BoxAtlas( xrange = (location, L), 
                              yrange = (0, W), 
                              zrange = (0, H))
        
        # Now we can make multiatlases with those as regions!
        # We know the lead-in and lead-out are plain Py, so add them.
        bareSpots.extend([leadin, leadout])

        worldAtlas = M.MultiAtlas( name = "WorldAtlas", atlases = goldSpots + bareSpots )

        # Mesh, energies...
        M.RectangularMesh( atlas = world, cellsize = (5e-9, 5e-9, 10e-9) )
        M.Demag()
        M.UniformExchange( A = 13e-12 )

        # Build in the Oe field, if there is one
        if Hmult != 0:
            # Build a list of fields over gold spots
            fields = [(atlas, 
                       (0, signMult * Hmult * u0, 0)) for atlas in goldSpots]
            # Supply that list of fields, with a default zero value
            # This takes care of putting field over hold and nowhere else
            fieldVec = M.AtlasVectorField( atlas = worldAtlas, 
                                           default_value = (0, 0, 0),
                                           values = fields )
            # Now that we've built that vector field, supply it to 
            # Oxs_FixedZeeman and we're done
            M.FixedZeeman( field = fieldVec, multiplier = MULTIPLIER )

        # Calculate u in a similar process - build up a multiatlas of the
        # gold and Py regions, with appropriate values bound to each one.
        # Another valid tactic would be building a 'gold' multiatlas
        # and a 'py' multiatlas, but this way demonstrates more Pythonicity.
        up = []
        for atlas in goldSpots:
            up.append([atlas, u0 * umult])
        for atlas in bareSpots:
            up.append([atlas, u0])
        # Build a scalar field trivially from the list
        uScalar = M.AtlasScalarField( atlas = worldAtlas, values = up )

        # Also, write an evolver.
        M.SpinTEvolve( alpha = 0.02, beta = 0.03, fixed_spins = (edges, 
                                                                 (left, right)), 
                                                                 u = uScalar )

        # Ready saturation, initial magnetization - these are uniform,
        # since the presence of gold is virtualized around the Py.
        satField = M.UniformScalarField( value = 800e3 )
        initMag  = M.FileVectorField( file="initial.omf", 
                                      atlas = worldAtlas, norm = 1 )

        # Set up the driver!
        M.TimeDriver( stopping_time = 6e-11, stage_count = 400, 
                      Ms = satField, m0 = initMag )

        # Outputs
        M.Destination( label = "arc", type = "mmArchive" )
        M.Schedule( output = "Oxs_TimeDriver:TimeDriver:Magnetization", 
                    label = "arc", stage = 1 )
        M.Schedule( output = "DataTable", label = "arc", step = 10 )

# Now we can go to town on this project.

# First, without field
for percentCurrentDrop in range(2, 23, 1):
    # Sorry, no floats in range calls - we need to use integers
    # and divide.
    uDrop = percentCurrentDrop / 100.0
    quickMIF(0, uDrop, 0, "NoOeField")

# Next, forward field and reverse field
for percentCurrentDrop in range(2, 23, 1):
    for fieldPerCurrentStep in range(1, 31, 1):
        uDrop = percentCurrentDrop/100.0
        fieldPerCurrent = fieldPerCurrentStep * 0.0276
        quickMIF(fieldPerCurrent, uDrop, 1, "FieldAntiparallel")
        quickMIF(fieldPerCurrent, uDrop, -1, "FieldParallel") 

# That's it! You should see the size of the output compared to the script.