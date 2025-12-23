NTH (aka NTW) frame:
- a_n: normal to velocity vector, pointed outwards
- a_t: component along instantaneous velocity vector
- a_h: component along osculating angmom direction

LVLH frame:
- x parallel to position vector, facing outwards
- z parallel to angular momentum vector
- y satisfies RHR, tangent to orbit path (not necessarily velocity vector)

ECI frame: Earth-centered J2000 frame as defined by SPICE

Orbital elements
- Semi-major axis: sma
- Eccentricity: ecc
- Inclination: inc
- Right ascension of ascending node: raan
- Argument of periapsis: aop
- True anomaly: ta or nu
- Mean anomaly: ma or M
- Eccentric anomaly: ea or E

- Argument of latitude at epoch: aol or aop + ta
- Longitude of periapsis: lop or raan + aop
- True longitude at epoch: tl or raan + aop + ta

- Eccentric longitude: F = raan + aop + E

Other angles:
- alpha: in plane pitch thrust direction steering angle, measured from 
velocity vector to the projection of the thrust vector onto the orbital plane
- delta: angle from local horizon to project of thrust on orbit plane
- gamma: FPA, angle from local horizon to velocity vector