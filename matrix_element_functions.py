# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:22:44 2019

@author: Oskari

This file contains functions used in the data analysis for characterizing
the molecular beam source in CeNTREX.

"""

#Import packages
import numpy as np
from tqdm import tqdm_notebook
import glob
from sympy.physics.wigner import wigner_3j, wigner_6j, clebsch_gordan
import sys
sys.path.append('../molecular-state-classes-and-functions/')
from classes import CoupledBasisState
from sympy import S
from datetime import datetime, timedelta


#### Functions
#%%
##### Calculating molecular beam intensity ####
#Function for evaluating scattering cross section given
def calculate_cross_section(ground_state, excited_state, B_state_eigenstates, gamma, detuning, wavelength):
    
    #Find branching ratio needed in calculating the cross section
    branching_ratio = calculate_branching_ratio(ground_state, excited_state, B_state_eigenstates)
    
    #Calculate the cross section
    F = float(ground_state.F)
    Fprime = float(excited_state.F)
    cross_section = wavelength**2/(2*np.pi)*(2*Fprime+1)/(2*F+1)*branching_ratio*(gamma/2)**2/((gamma/2)**2 + detuning**2)
    
    return float(cross_section)


#Defining a utility function that can be used to turns floats into rational numbers in sympy
def rat(number):
    return S(str(number),rational = True)


#Function for evaluation the electric dipole matrix element between a ground state and excited state
def calculate_microwave_ED_matrix_element(ground_state, excited_state,reduced = True, pol_vec = np.array((0,0,1))):
    #Find quantum numbers for ground state
    J = float(ground_state.J)
    F1 = float(ground_state.F1)
    F = float(ground_state.F)
    mF = float(ground_state.mF)
    I1 = float(ground_state.I1)
    I2 = float(ground_state.I2)
    
    #Find quantum numbers of excited state
    Jprime = float(excited_state.J)
    F1prime = float(excited_state.F1)
    Fprime = float(excited_state.F)
    mFprime = float(excited_state.mF)
    
    #Calculate reduced matrix element
    M_r = (np.sqrt(float((2*F1+1) * (2*F1prime+1) * (2*F+1)* (2*Fprime+1))) * float(wigner_6j(Jprime, F1prime,1/2,F1,J,1))
               * float(wigner_6j(F1prime, Fprime,1/2,F,F1,1)) * np.sqrt(float((2*J+1) * (2*Jprime+1))) 
               *(-1)**(F1prime+J+Fprime+F1+1)
               * float(wigner_3j(J,1,Jprime,0,0,0) * (-1)**J))
    
    if reduced:
        return float(M_r)
    else:
        p_vec = {}
        p_vec[-1] = -1/np.sqrt(2) * (pol_vec[0] + 1j *pol_vec[1])
        p_vec[0] = pol_vec[2]
        p_vec[1] = +1/np.sqrt(2) * (pol_vec[0] - 1j *pol_vec[1])
        
        prefactor = 0
        for p in range(-1,2):
            prefactor +=  (-1)**(p+F-mF) * p_vec[p] *  float(wigner_3j(F,1,Fprime,-mF,-p,mFprime))
        
        
        return prefactor*float(M_r)
    
    
#Function for evaluation the electric dipole matrix element between a ground state and excited state in uncoupled basis
def calculate_microwave_ED_matrix_element_uncoupled(ground_state, excited_state,reduced = True, pol_vec = np.array((0,0,1))):
    #Find quantum numbers for ground state
    J = float(ground_state.J)
    mJ = float(ground_state.mJ)
    I1 = float(ground_state.I1)
    m1 = float(ground_state.m1)
    I2 = float(ground_state.I2)
    m2 = float(ground_state.m2)
    
    #Find quantum numbers of excited state
    Jprime = float(excited_state.J)
    mJprime = float(excited_state.mJ)
    I1prime = float(excited_state.I1)
    m1prime = float(excited_state.m1)
    I2prime = float(excited_state.I2)
    m2prime = float(excited_state.m2)
    
    #Calculate reduced matrix element
    M_r = (wigner_3j(J,1,Jprime,0,0,0) * np.sqrt((2*J+1)*(2*Jprime+1)) 
            * float(I1 == I1prime and m1 == m1prime 
                    and I2 == I2prime and m2 == m2prime))
    
    
    
    if reduced:
        return float(M_r)
    else:
        p_vec = {}
        p_vec[-1] = -1/np.sqrt(2) * (pol_vec[0] + 1j *pol_vec[1])
        p_vec[0] = pol_vec[2]
        p_vec[1] = +1/np.sqrt(2) * (pol_vec[0] - 1j *pol_vec[1])
        
        prefactor = 0
        for p in range(-1,2):
            prefactor +=  (-1)**(p-mJ) * p_vec[p] *  float(wigner_3j(J,1,Jprime,-mJ,-p,mJprime))
        
        
        return prefactor*float(M_r)

#Function for evaluating the electric dipole matrix element between a superposition state (excited state) and one of the
#hyperfine states of the ground state. I'm only calculating the angular part here since that is all that is needed for the 
#branching ratio and scattering cross section calculations
def calculate_microwave_ED_matrix_element_superposition(ground_state, excited_state, X_state_eigenstates):
    #Find quantum numbers for ground state
    J = ground_state.J
    F1 = ground_state.F1
    F = ground_state.F
    I1 = ground_state.I1
    I2 = ground_state.I2
    
    #Find quantum numbers of excited state and determine what the 'real' mixed eigenstate is by looking it up from a 
    #dictionary
    Jprime = excited_state.J
    F1prime = excited_state.F1
    Fprime = excited_state.F
    Pprime = excited_state.P
    
    
    #Generate the name of the state
    ground_state_name = "|J = %s, F1 = %s, F = %s, mF = 0, I1 = %s, I2 = %s>"%(rat(J),rat(F1),rat(F)
                                                                                ,rat(I1),rat(I2))
    excited_state_name = "|J = %s, F1 = %s, F = %s, mF = 0, I1 = %s, I2 = %s>"%(rat(Jprime),rat(F1prime),rat(Fprime)
                                                                                ,rat(I1),rat(I2))
    
    #Find states in dictionary
    ground_state_mixed = X_state_eigenstates[ground_state_name]
    excited_state_mixed = X_state_eigenstates[excited_state_name]
        
    #Calculate reduced matrix elements for each component of the excited state and sum them together to get the
    #total reduced matrix element
    M_r = 0
    
    for amp1, basis_state1 in ground_state_mixed.data:
        for amp2, basis_state2 in excited_state_mixed.data:
            M_r += amp1*np.conjugate(amp2)*calculate_microwave_ED_matrix_element(basis_state1, basis_state2)
        
    return M_r


#Function for evaluating the electric dipole matrix element between a superposition state (excited state) and one of the
#hyperfine states of the ground state. I'm only calculating the angular part here since that is all that is needed for the 
#branching ratio and scattering cross section calculations
def calculate_microwave_ED_matrix_element_mixed_state(ground_state, excited_state,reduced = True,pol_vec = np.array((0,0,1))):        
    #Calculate reduced matrix elements for each component of the excited state and sum them together to get the
    #total reduced matrix element
    M = 0
    
    for amp1, basis_state1 in ground_state.data:
        for amp2, basis_state2 in excited_state.data:
            M += amp1*np.conjugate(amp2)*calculate_microwave_ED_matrix_element(basis_state1, basis_state2,reduced,pol_vec)
        
    return M


#Function for evaluating the electric dipole matrix element between a superposition state (excited state) and one of the
#hyperfine states of the ground state. I'm only calculating the angular part here since that is all that is needed for the 
#branching ratio and scattering cross section calculations
def calculate_microwave_ED_matrix_element_mixed_state_uncoupled(ground_state, excited_state,reduced = True,pol_vec = np.array((0,0,1))):        
    #Calculate reduced matrix elements for each component of the excited state and sum them together to get the
    #total reduced matrix element
    M = 0
    
    for amp1, basis_state1 in ground_state.data:
        for amp2, basis_state2 in excited_state.data:
            M += amp1*np.conjugate(amp2)*calculate_microwave_ED_matrix_element_uncoupled(basis_state1, basis_state2,reduced,pol_vec)
        
    return M

### Matrix element functions for OBE integrator ### 
from sympy.physics.wigner import wigner_3j, wigner_6j

def threej_f(j1,j2,j3,m1,m2,m3):
    return complex(wigner_3j(j1,j2,j3,m1,m2,m3))

def sixj_f(j1,j2,j3,j4,j5,j6):
    return complex(wigner_6j(j1,j2,j3,j4,j5,j6))


def ED_ME_coupled(bra,ket, pol_vec = np.array([1,1,1]), rme_only = False):
    """
    Function for calculating electric dipole matrix elements between CoupledBasisStates.
    
    inputs:
    ket = CoupledBasisState object
    bra = CoupledBasisState object
    pol_vec = polarization vector for the light that is driving the transition (the default is useful when calculating branching ratios)
    rme_only = True if want to return reduced matrix element, False if want angular component also
    
    returns:
    ME = (reduced) electric dipole matrix element between ket and bra
    """
    
    #Find quantum numbers for ground state
    F = bra.F
    mF = bra.mF
    J = bra.J
    F1 = bra.F1
    I1 = bra.I1
    I2 = bra.I2
    Omega = bra.Omega
    
    #Find quantum numbers for excited state
    Fp = ket.F
    mFp = ket.mF
    Jp = ket.J
    F1p = ket.F1
    I1p = ket.I1
    I2p = ket.I2
    Omegap = ket.Omega
    
    #Calculate the reduced matrix element
    q = Omega - Omegap
    ME = ((-1)**(F1+J+Fp+F1p+I1+I2) * np.sqrt((2*F+1)*(2*Fp+1)*(2*F1p+1)*(2*F1+1)) * sixj_f(F1p,Fp,I2,F,F1,1) 
          * sixj_f(Jp,F1p,I1,F1,J,1) * (-1)**(J-Omega) *np.sqrt((2*J+1)*(2*Jp+1)) * threej_f(J,1,Jp,-Omega, q, Omegap)
          * float(np.abs(q) < 2))
    
    #If we want the complete matrix element, calculate angular part
    if not rme_only:
        
        #Calculate elements of the polarization vector in spherical basis
        p_vec = {}
        p_vec[-1] = -1/np.sqrt(2) * (pol_vec[0] + 1j *pol_vec[1])
        p_vec[0] = pol_vec[2]
        p_vec[1] = +1/np.sqrt(2) * (pol_vec[0] - 1j *pol_vec[1])
        
        #Calculate the value of p that connects the states
        p = mF-mFp
        p = p*int(np.abs(p) <= 1)
        #Multiply RME by the angular part
        ME = ME * (-1)**(F-mF) * threej_f(F,1,Fp, -mF, p, mFp) * p_vec[p] * int(np.abs(p) <= 1)
    
    #Return the matrix element
    return ME

#Function for evaluation the electric dipole matrix element between a ground state and excited state in uncoupled basis
def ED_ME_uncoupled(bra,ket, pol_vec = np.array([1,1,1]), rme_only = False):
    #Find quantum numbers for ground state
    J = bra.J
    mJ = float(bra.mJ)
    I1 = bra.I1
    m1 = bra.m1
    I2 = bra.I2
    m2 = bra.m2
    
    #Find quantum numbers of excited state
    Jprime = ket.J
    mJprime = ket.mJ
    I1prime = ket.I1
    m1prime = ket.m1
    I2prime = ket.I2
    m2prime = ket.m2
    
    #Calculate reduced matrix element
    M_r = (threej_f(J,1,Jprime,0,0,0) * np.sqrt((2*J+1)*(2*Jprime+1)) 
            * float(I1 == I1prime and m1 == m1prime 
                    and I2 == I2prime and m2 == m2prime))
    
    
    #If reduced matrix element is desired, return that. Otherwise calculate the angular prefactor
    #for the matrix element.
    if rme_only:
        return M_r
    else:
        p_vec = {}
        p_vec[-1] = -1/np.sqrt(2) * (pol_vec[0] + 1j *pol_vec[1])
        p_vec[0] = pol_vec[2]
        p_vec[1] = +1/np.sqrt(2) * (pol_vec[0] - 1j *pol_vec[1])
        
        prefactor = 0
        for p in range(-1,2):
            prefactor +=  (-1)**(p-mJ) * p_vec[p] *  threej_f(J,1,Jprime,-mJ,-p,mJprime)
        
        
        return prefactor*M_r

#Function for calculating matrix elements between states that are superpositions
#in coupled basis
def ED_ME_mixed_state(bra, ket, pol_vec = np.array([1,1,1]), reduced = False):
    """
    Calculates electric dipole matrix elements between mixed states

    inputs:
    bra = state object
    ket = state object
    pol_vec = polarization vector for the light that is driving the transition (the default is useful when calculating branching ratios)

    outputs:
    ME = matrix element between the two states
    """
    ME = 0
    bra = bra.transform_to_omega_basis()
    ket = ket.transform_to_omega_basis()
    for amp_bra, basis_bra in bra.data:
        for amp_ket, basis_ket in ket.data:
            ME += amp_bra.conjugate()*amp_ket*ED_ME_coupled(basis_bra, basis_ket, pol_vec = pol_vec, rme_only = reduced)

    return ME

#Function for calculating matrix elements between states that are superpositions
#in coupled basis
def ED_ME_mixed_state_uc(bra, ket, pol_vec = np.array([1,1,1]), reduced = False):
    """
    Calculates electric dipole matrix elements between mixed states

    inputs:
    bra = state object
    ket = state object
    pol_vec = polarization vector for the light that is driving the transition (the default is useful when calculating branching ratios)

    outputs:
    ME = matrix element between the two states
    """
    ME = 0
    bra = bra.transform_to_omega_basis()
    ket = ket.transform_to_omega_basis()
    for amp_bra, basis_bra in bra.data:
        for amp_ket, basis_ket in ket.data:
            ME += amp_bra.conjugate()*amp_ket*ED_ME_uncoupled(basis_bra, basis_ket, pol_vec = pol_vec, rme_only = reduced)

    return ME
    