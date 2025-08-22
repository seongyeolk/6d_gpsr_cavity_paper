import numpy as np
import torch
from bmadx import PI
from bmadx.bmad_torch.track_torch import (
    TorchCrabCavity,
    TorchDrift,
    TorchLattice,
    TorchQuadrupole,
    TorchRFCavity,
    TorchSBend,
    TorchSextupole,
)


def quad_drift(l_d=1.0, l_q=0.1, n_slices=5):
    """Creates quad + drift lattice

    Params
    ------
        l_d: float
            drift length (m). Default: 1.0

        l_q: float
            quad length (m). Default: 0.1

        n_steps: int
            slices in quad tracking. Default: 5

    Returns
    -------
        lattice: bmad_torch.TorchLattice
            quad scan lattice
    """

    q1 = TorchQuadrupole(torch.tensor(l_q), torch.tensor(0.0), n_slices)
    d1 = TorchDrift(torch.tensor(l_d))
    lattice = TorchLattice([q1, d1])

    return lattice


def sextupole_drift(l_d=1.0, l_q=0.1, n_slices=5):
    """Creates quad + drift lattice

    Params
    ------
        l_d: float
            drift length (m). Default: 1.0

        l_q: float
            quad length (m). Default: 0.1

        n_steps: int
            slices in quad tracking. Default: 5

    Returns
    -------
        lattice: bmad_torch.TorchLattice
            quad scan lattice
    """

    q1 = TorchSextupole(torch.tensor(l_q), torch.tensor(0.0), n_slices)
    d1 = TorchDrift(torch.tensor(l_d))
    lattice = TorchLattice([q1, d1])

    return lattice


def quad_tdc_bend(
    p0c,
    l_q = 0.11,
    l_tdc = 0.01,
    f_tdc = 1.3e9,
    phi_tdc = 0.0,
    l_bend = 0.3018,
    theta_on = - 20.0 * PI / 180.0,
    l1 = 0.790702,
    l2 = 0.631698,
    l3 = 0.889,
    dipole_on=False
):
    """
    Creates diagnostic lattice with quad, tdc and bend. 
    Default values are for AWA Zone 5.
    
    Params
    ------
        p0c: float
            design momentum (eV/c). 
            
        l_q: float
            quad length (m). Default: 0.11
        
        l_tdc: float
            TDC length (m). NOTE: for now, Bmad-X TDC is a single kick at the
            TDC center, so this length doesn't change anything. Default: 0.01
        
        f_tdc: float
            TDC frequency (Hz). Default: 1.3e9
        
        phi_tdc: float
            TDC phase (rad). 0.0 corresponds to zero crossing phase with 
            positive slope. Default: 0.0
            
        l_bend: float
            Bend length (m). Default: 0.3018
            
        theta_on: float
            Bending angle when bending magnet is on (rad). Negative angle deflects
            in +x direction. Default: -20*pi/180
        
        l1: float
            Center-to-center distance between quad and TDC (m). Default: 0.790702
        
        l2: float
            Center-to-center distance between TDC and dipole (m). Default: 0.631698
        
        l3: float
            Distance from screens to dipole center (m). Default: 0.889
            
        dipole_on: bool
            Initializes the lattice with dipole on or off. Default: False
            
    Returns
    -------
        TorchLattice
    """
    
    # initialize dipole params when on/off:
    if dipole_on:
        theta = theta_on # negative deflects in +x
        l_arc = l_bend * theta / np.sin(theta)
        g = theta / l_arc
    if not dipole_on:
        g = -2.22e-16  # machine epsilon to avoid numerical error
        theta = np.arcsin(l_bend * g)
        l_arc = theta / g

    # Drifts with geometrical corrections:

    # Drift from Quad to TDC (0.5975)
    l_d1 = l1 - l_q / 2 - l_tdc / 2

    # Drift from TDC to Bend (0.3392)
    l_d2 = l2 - l_tdc / 2 - l_bend / 2

    # Drift from Bend to YAG 2 (corrected for dipole on/off)
    l_d3 = l3 - l_bend / 2 / np.cos(theta)

    # Elements:
    q = TorchQuadrupole(L=torch.tensor(l_q), K1=torch.tensor(0.0), NUM_STEPS=5)

    d1 = TorchDrift(L=torch.tensor(l_d1))

    tdc = TorchCrabCavity(
        L=torch.tensor(l_tdc),
        VOLTAGE=torch.tensor(0.0),
        RF_FREQUENCY=torch.tensor(f_tdc),
        PHI0=torch.tensor(phi_tdc),
        TILT=torch.tensor(PI / 2),
    )

    d2 = TorchDrift(L=torch.tensor(l_d2))

    bend = TorchSBend(
        L=torch.tensor(l_arc),
        P0C=torch.tensor(p0c),
        G=torch.tensor(g),
        E1=torch.tensor(0.0),
        E2=torch.tensor(theta),
        FRINGE_AT="both_ends",
        FRINGE_TYPE="linear_edge"
    )

    d3 = TorchDrift(L=torch.tensor(l_d3))

    lattice = TorchLattice([q, d1, tdc, d2, bend, d3])

    return lattice


# PALXFEL_BC1 for experimental demonstration
def palxfel_Simulation(
    p0c,
    l_qt = 0.065,
    l_bend = 0.20,
    theta_on = 4.97 * PI / 180.0,
    D1l = 3.82431,
    D2l = 0.22145,
    D3l = 1.69650-0.6, # Where 0.6 is the length of the X-linearizer
    D4l = 0.91000,
    D5l = 0.31750,
    D6l = 1.00446,
    D61l= 0.10000,
    D62l= 3.19696,
    D7l = 0.67800,
    D8l = 0.32200,
    D9l = 4.30142,
    dipole_on=True
):
    
    """
    Returns
    -------
        TorchLattice
    """
    
    # Elements:
    L1_Q3 = TorchQuadrupole(L=torch.tensor(l_qt), K1=torch.tensor(0.0), NUM_STEPS=5)
    
    # Elegant simulation setting
    BC1_Q1= TorchQuadrupole(L=torch.tensor(l_qt), K1=torch.tensor(0.0), NUM_STEPS=2)
    BC1_Q2= TorchQuadrupole(L=torch.tensor(l_qt), K1=torch.tensor(0.0), NUM_STEPS=2)
    BC1_Q3= TorchQuadrupole(L=torch.tensor(l_qt), K1=torch.tensor(0.0), NUM_STEPS=2)
    
    
    # X-Linearizer
    XLIN = TorchRFCavity(
        L=torch.tensor(0.600),
        VOLTAGE=torch.tensor(13.5E6),
        RF_FREQUENCY=torch.tensor(11.424E9),
        PHI0=torch.tensor(0.0),
        TILT=torch.tensor(0))

    
    XLIN2   = TorchDrift(L=torch.tensor(0.6))
    D1_Drif = TorchDrift(L=torch.tensor(D1l))
    D2_Drif = TorchDrift(L=torch.tensor(D2l))
    D3_Drif = TorchDrift(L=torch.tensor(D3l))
    D4_Drif = TorchDrift(L=torch.tensor(D4l))
    D5_Drif = TorchDrift(L=torch.tensor(D5l))
    D6_Drif = TorchDrift(L=torch.tensor(D6l))
    D61_Drif= TorchDrift(L=torch.tensor(D61l))
    D62_Drif= TorchDrift(L=torch.tensor(D62l))
    D7_Drif = TorchDrift(L=torch.tensor(D7l))
    D8_Drif = TorchDrift(L=torch.tensor(D8l))
    D9_Drif = TorchDrift(L=torch.tensor(D9l))
    
    # initialize dipole params when on/off:
    if dipole_on:
        theta = theta_on # negative deflects in +x
        l_arc = l_bend
        g = theta / l_arc
    if not dipole_on:
        g = -2.22e-16  # machine epsilon to avoid numerical error
        theta = np.arcsin(l_bend * g)
        l_arc = theta / g

    # Elements:
    BC1_B1A = TorchSBend(
        L=torch.tensor(l_arc),
        P0C=torch.tensor(p0c),
        G=torch.tensor(-g),
        E1=torch.tensor(0.0),
        E2=torch.tensor(-theta),
        FINT=torch.tensor(0.5),
        HGAP=torch.tensor(1.500000E-02),
        FRINGE_AT="both_ends",
        FRINGE_TYPE="linear_edge"
    )

    BC1_B2A = TorchSBend(
        L=torch.tensor(l_arc),
        P0C=torch.tensor(p0c),
        G=torch.tensor(g),
        E1=torch.tensor(theta),
        E2=torch.tensor(0.0),
        FINT=torch.tensor(0.5),
        HGAP=torch.tensor(1.500000E-02),
        FRINGE_AT="both_ends",
        FRINGE_TYPE="linear_edge"
    )

    # Elements:
    BC1_B1A2 = TorchDrift(L=torch.tensor(0.2))
    BC1_B2A2 = TorchDrift(L=torch.tensor(0.2))
    ###############
    
    
    
    lattice_SCM1= TorchLattice([L1_Q3, D1_Drif, BC1_Q1, D2_Drif, 
                                XLIN, D3_Drif, BC1_Q2, D4_Drif, BC1_Q3,
                                D5_Drif, 
                                BC1_B1A2, D6_Drif, D61_Drif, D62_Drif, BC1_B2A2,
                                D7_Drif])
    
    lattice_SCM2= TorchLattice([L1_Q3, D1_Drif, BC1_Q1, D2_Drif, 
                                XLIN, D3_Drif, BC1_Q2, D4_Drif, BC1_Q3,
                                D5_Drif, 
                                BC1_B1A, D6_Drif, D61_Drif, D62_Drif, BC1_B2A,
                                D7_Drif])
    
    return lattice_SCM1, lattice_SCM2
