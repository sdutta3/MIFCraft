### IGNORE THIS PART ###

import sys
sys.path.append("..")

###  STOP  IGNORING  ###

from mifcraft import MIF
from math import pi
Mu0 = pi * 4e-7

for field in range(20, 200, 20):
    with MIF(filename='mifs/Field' + str(field) + '.mif',
             basename='../data/Field' + str(field) + '/out') as M:
        
        Brick = M.BoxAtlas( xrange = (0, 100e-9),
                            yrange = (0, 100e-9), 
                            zrange = (0, 12e-9))
        
        M.RectangularMesh( atlas = Brick, cellsize = (4e-9, 4e-9, 4e-9) )
        M.UZeeman( multiplier = 79, Hrange = [(0, 0, field, 0, 0, field, 1)] )
        M.UniformExchange( A = 13e-12 )
        M.Demag()
        
        satMag = M.UniformScalarField( value = 800e3 )
        startMag = M.RandomVectorField( min_norm = 1, max_norm = 1 )
        
        M.RungeKuttaEvolve( )
        M.TimeDriver( stopping_dm_dt = 0.1,
                      name = "TimeDriver",
                      Ms = satMag, m0 = startMag )
        
        M.Destination( label = "watch", type = "mmDisp" )
        M.Schedule( label = "watch", step = 20, output = "Oxs_TimeDriver:TimeDriver:Magnetization")
        