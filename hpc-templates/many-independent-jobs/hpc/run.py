"""
FILE: run.py
PURPOSE: Run a batch of data through user-defined processing.
DESCRIPTION:

"""
import os
import pickle as pkl
from mpi4py import MPI
from typing import Union, Optional
import sys
import importlib

run_single = importlib.import_module(f"hpc.{os.environ['RUN_NAME']}.run_single").run_single

MPIComm = Union[MPI.Intracomm, MPI.Intercomm]
    
def main(
    job_id: int, 
    input_dir: str, 
    output_dir: str, 
):
    """Main entry point.

    Note that job_id is the job array task ID, not the SLURM job ID.
    """
    if ("OMPI_COMM_WORLD_RANK" not in os.environ) | ("OMPI_COMM_WORLD_SIZE" not in os.environ):
        sys.exit("Program exited. Required environment variables are missing. OMPI_COMM_WORLD_RANK is {} and OMPI_COMM_WORLD_SIZE is {}.".format(os.environ.get("OMPI_COMM_WORLD_RANK"), os.environ.get("OMPI_COMM_WORLD_SIZE")))
    else:
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK"))
        size = int(os.environ.get("OMPI_COMM_WORLD_SIZE"))
    _run(job_id, rank, input_dir, output_dir)

def _run(
    job_id: str, 
    batch_id: str,
    input_dir: str,
    output_dir: str,
):
    # Load data batch
    data_batch_filepath = os.path.join(input_dir, f'data_batch_{job_id}_{batch_id}.pkl')
    with open(data_batch_filepath, 'rb') as f:
        ids_and_data_batch = pkl.load(f)
    if len(ids_and_data_batch) == 0:
        result = []
        ids = []
    else:
        ids, data_batch = list(zip(*ids_and_data_batch))  # unzip

        # Run the batch through processing
        result = _run_batch(MPI.COMM_SELF, job_id, batch_id, data_batch)
    
    # Save results
    results_batch_filepath = os.path.join(output_dir, f'results_{job_id}_{batch_id}.pkl')
    os.makedirs(output_dir, exist_ok=True)
    with open(results_batch_filepath, 'wb') as f:
        pkl.dump(list(zip(ids, result)), f)   

def _run_batch(
    mpi_comm: MPIComm,
    job_id: int,
    batch_id: int,
    data_batch: list,
):
    """Run a batch of data through user-defined processing.

    Returns a list of same length as `data_batch` containing the result for each batch.
    
    Parameters
    ----------
    mpi_comm: MPI communicator
    job_id: int
        Job array rask ID
    batch_id: int
    data_batch: list
        List of data to run
    
    Returns
    -------
    results: list
        List of results
    """
    results = []
    for data in data_batch:
        results.append(run_single(data))
    return results


if __name__=='__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', type=str)
    parser.add_argument('-i', '--input-dir', type=str)
    parser.add_argument('-o', '--output-dir', type=str)
    args = vars(parser.parse_args())

    t1 = time.time()
    main(
        job_id=args['job_id'], 
        input_dir=args['input_dir'],
        output_dir=args['output_dir'],
    )
    t2 = time.time()
    print('Total run time: {:.5f}'.format(t2 - t1))