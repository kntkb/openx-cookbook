#!/usr/bin/env python
# coding: utf-8
import os, sys, math
import numpy as np
import glob
import click
import mdtraj
import logging
import netCDF4 as nc
import warnings
import pandas as pd
import logging
import openmmtools as mmtools
from pymbar import timeseries
from openmm import *
from openmm.app import *
import barnaba as bb
from barnaba import definitions
from barnaba.nucleic import Nucleic


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ==============================================================================
# Extract trajectory from NetCDF4 file
# https://github.com/choderalab/yank/blob/master/Yank/analyze.py
# ==============================================================================
def extract_trajectory(reference, nc_path, nc_checkpoint_file=None, checkpoint_interval=1, state_index=None, replica_index=None,
                       start_frame=0, end_frame=-1, skip_frame=1, keep_solvent=True):
    """Extract phase trajectory from the NetCDF4 file.
    Parameters
    ----------
    reference : str
        Path to reference pdb file to extract topology information
    nc_path : str
        Path to the primary nc_file storing the analysis options
    nc_checkpoint_file : str or None, Optional
        File name of the checkpoint file housing the main trajectory
        Used if the checkpoint file is differently named from the default one chosen by the nc_path file.
        Default: None
    checkpoint_interval : int >= 1, Default: 1
        The frequency at which checkpointing information is written relative to analysis information.
        This is a multiple of the iteration at which energies is written, hence why it must be greater than or equal to 1.
        Checkpoint information cannot be written on iterations which where ``iteration % checkpoint_interval != 0``.
    state_index : int, optional
        The index of the alchemical state for which to extract the trajectory.
        One and only one between state_index and replica_index must be not None
        (default is None).
    replica_index : int, optional
        The index of the replica for which to extract the trajectory. One and
        only one between state_index and replica_index must be not None (default
        is None).
    start_frame : int, optional
        Index of the first frame to include in the trajectory (default is 0).
    end_frame : int, optional
        Index of the last frame to include in the trajectory. If negative, will
        count from the end (default is -1).
    skip_frame : int, optional
        Extract one frame every skip_frame (default is 1).
    keep_solvent : bool, optional
        If False, solvent molecules are ignored (default is True).
    Returns
    -------
    trajectory: mdtraj.Trajectory
        The trajectory extracted from the netcdf file.
    """
    # Check correct input
    if (state_index is None) == (replica_index is None):
        raise ValueError('One and only one between "state_index" and '
                         '"replica_index" must be specified.')
    if not os.path.isfile(nc_path):
        raise ValueError('Cannot find file {}'.format(nc_path))

    # Import simulation data
    reporter = None
    try:
        reporter = mmtools.multistate.MultiStateReporter(nc_path, open_mode='r', checkpoint_storage=nc_checkpoint_file, checkpoint_interval=checkpoint_interval)
        reference = mdtraj.load_pdb(reference)
        topology = reference.topology
        
        # Get dimensions
        # Assume full iteration until proven otherwise
        last_checkpoint = True
        trajectory_storage = reporter._storage_checkpoint
        if not keep_solvent:
            # If tracked solute particles, use any last iteration, set with this logic test
            full_iteration = len(reporter.analysis_particle_indices) == 0
            if not full_iteration:
                trajectory_storage = reporter._storage_analysis
                topology = topology.subset(reporter.analysis_particle_indices)

        n_iterations = reporter.read_last_iteration(last_checkpoint=last_checkpoint)
        n_frames = trajectory_storage.variables['positions'].shape[0]
        n_atoms = trajectory_storage.variables['positions'].shape[2]
        logger.info('Number of frames: {}, atoms: {}'.format(n_frames, n_atoms))

        # Determine frames to extract.
        # Convert negative indices to last indices.
        if start_frame < 0:
            start_frame = n_frames + start_frame
        if end_frame < 0:
            end_frame = n_frames + end_frame + 1
        frame_indices = range(start_frame, end_frame, skip_frame)
        if len(frame_indices) == 0:
            raise ValueError('No frames selected')
        logger.info('Extracting frames from {} to {} every {}'.format(start_frame, end_frame, skip_frame))


        # Determine the number of frames that the trajectory will have.
        if state_index is None:
            n_trajectory_frames = len(frame_indices)
        else:
            # With SAMS, an iteration can have 0 or more replicas in a given state.
            # Deconvolute state indices.
            state_indices = [None for _ in frame_indices]
            for i, iteration in enumerate(frame_indices):
                replica_indices = reporter._storage_analysis.variables['states'][iteration, :]
                state_indices[i] = np.where(replica_indices == state_index)[0]
                #print(state_index, replica_indices, np.where(replica_indices == state_index)[0])
            n_trajectory_frames = sum(len(x) for x in state_indices)

        # Initialize positions and box vectors arrays.
        # MDTraj Cython code expects float32 positions.
        positions = np.zeros((n_trajectory_frames, n_atoms, 3), dtype=np.float32)
        box_vectors = np.zeros((n_trajectory_frames, 3, 3), dtype=np.float32)

        # Extract state positions and box vectors.
        if state_index is not None:
            logger.info('Extracting positions of state {}...'.format(state_index))

            # Extract state positions and box vectors.
            frame_idx = 0
            for i, iteration in enumerate(frame_indices):
                for replica_index in state_indices[i]:
                    positions[frame_idx, :, :] = trajectory_storage.variables['positions'][iteration, replica_index, :, :].astype(np.float32)
                    box_vectors[frame_idx, :, :] = trajectory_storage.variables['box_vectors'][iteration, replica_index, :, :].astype(np.float32)
                    frame_idx += 1

        else:  # Extract replica positions and box vectors
            logger.info('Extracting positions of replica {}...'.format(replica_index))

            for i, iteration in enumerate(frame_indices):
                positions[i, :, :] = trajectory_storage.variables['positions'][iteration, replica_index, :, :].astype(np.float32)
                box_vectors[i, :, :] = trajectory_storage.variables['box_vectors'][iteration, replica_index, :, :].astype(np.float32)
    finally:
        if reporter is not None:
            reporter.close()

    # Create trajectory object
    logger.info('Creating trajectory object...')
    trajectory = mdtraj.Trajectory(positions, topology)
    trajectory.unitcell_vectors = box_vectors

    # Export trajectory (overwrite exisiting file)
    #trajectory.save_netcdf('md.nc')
    #trajectory[-1].save_amberrst7('md.rst7')
    trajectory[-1].save_pdb('./pdb/replica{}.pdb'.format(replica_index))

    return trajectory



def run(kwargs):
    input_prefix = kwargs['input_prefix']
    ref_prefix = kwargs['ref_prefix']
    keep_solvent = kwargs['keep_solvent']
    sel_frame = kwargs['frame']


    """
    Load
    """
    # PDB and trajectory
    ref_pdb = os.path.join(ref_prefix, "min.pdb")    
    ncfile = os.path.join(input_prefix, "enhanced.nc")

    # Check number of states and replicas
    reporter = mmtools.multistate.MultiStateReporter(ncfile, open_mode='r')
    n_states = reporter.n_states
    n_replicas = reporter.n_replicas
    print(">n_states: {}".format(n_states))
    print(">n_replicas: {}".format(n_replicas))
    del reporter

    for index in range(0, n_replicas):
        print("processing replica {}....".format(index))
        # Extract trajectory by replicas
        # Returns either trajectory with full atoms or only specified atoms
        t = extract_trajectory(reference=ref_pdb, nc_path=ncfile, start_frame=sel_frame, end_frame=sel_frame, skip_frame=1, replica_index=index, keep_solvent=keep_solvent)
        

@click.command()
@click.option("--input_prefix", required=True, help="path to input file")
@click.option("--ref_prefix", required=True, help="path to reference files")
@click.option("--keep_solvent", default=True, help="keep solvent during analysis")
@click.option("--frame", default=-1, help="frame index to export snapshot")
def cli(**kwargs):
    run(kwargs)



if __name__ == '__main__':
    cli()

