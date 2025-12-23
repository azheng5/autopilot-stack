class Spacecraft:
    
    def __init__(self,
                 id: str,
                 wet_mass: float,
                 dry_mass: float,
                 Isp: float,
                 Cd: float,
                 A_ref: float) -> None:
        
        self.id = id
        self.wet_mass = wet_mass
        self.dry_mass = dry_mass
        self.Isp = Isp
        self.Cd = Cd
        self.A_ref = A_ref
        #TODO add term for constant thrust
        #TODO power, solar array degradation term, etc