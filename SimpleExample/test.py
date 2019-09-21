### IGNORE THIS PART ###

import sys
sys.path.append("..")

###  STOP  IGNORING  ###

# Simplest possible case: Let's randomize a brick and let it relax
# It'll be permalloy
from mifcraft import MIF
from math import pi

Mu0 = pi * 4e-7

with MIF(filename='test_1.mif', basename='test_1') as myMIF:
    Brick = myMIF.BoxAtlas( xrange = (0, 100e-9),
                            yrange = (0, 100e-9),
                            zrange = (0,  20e-9))
    myMIF.RectangularMesh( atlas = Brick,
                           cellsize = (5e-9, 5e-9, 5e-9) )
    myMIF.Demag()
    myMIF.UniformExchange( A = 13e-12 )
    myMIF.RungeKuttaEvolve( alpha = 0.5 )
 
    Ms = myMIF.UniformScalarField( value = 800e3 )
    m0 = myMIF.PlaneRandomVectorField( min_norm = 1, max_norm = 1, plane_normal = (0, 0, 1) )
    
    myMIF.TimeDriver( stopping_dm_dt = 0.1,
                      stage_count = 1,
                      Ms = Ms,
                      m0 = m0 )
    
    myMIF.Destination(label='disp', type='mmDisp')
    myMIF.Schedule(output='Oxs_TimeDriver:TimeDriver:Magnetization',
                   label='disp',
                   step=50)
    
    
    
    
