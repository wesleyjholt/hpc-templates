import os
from copy import deepcopy
from typing import Callable
import pyslurm
from .utils import load_yaml, check_has_keys, merge_dicts, parse_field, parse_slurm_array, estimate_total_time
from .split_data import main as split_data

class ManySmallJobs(dict):
    def __init__(
        self, 
        single_run_fn: Callable[[int, list, str], list], 
        config_files: list[str]
    ):
        configs = [load_yaml(f) for f in config_files]
        config = merge_dicts(configs)
        super().__init__(config)
        self.single_run_fn = single_run_fn
        self.check_config_is_valid()

        # The "main job"
        main_job_script_args = [
            f"--single-run-fn {self.single_run_fn}",
            f"--batched-data-dir {self.batched_data_dir}",
            f"--batched-results-dir {self.batched_results_dir}",
            f"--split-results-dir {self.split_results_dir}",
            f"--job-array {self.array_elements}"
        ]
        main_job_script_args = ' '.join([parse_field(arg) for arg in main_job_script_args])
        self.main_job = pyslurm.JobSubmitDescription(
            **self['main_slurm_args'],
            script='main',
            script_args=main_job_script_args
        )
        self.main_job_id = None

        # The "merge job"
        merge_job_script_args = [
            f"--batched-results-dir {self.batched_results_dir}",
            f"--merged-results-file {self.merged_results_file}",
            f"--tmp-dir {self['tmp_dir']}",
            f"--job-array {self.array_elements}",
            f"--ntasks-per-job {self['main_slurm_args']['ntasks']}"
        ]
        merge_job_script_args = ' '.join([parse_field(arg) for arg in merge_job_script_args])
        self.merge_job = pyslurm.JobSubmitDescription(
            **self['merge_slurm_args'],
            script='merge',
            script_args=merge_job_script_args
        )
        self.merge_job_id = None

    def check_config_is_valid(self):
        check_has_keys(self, required_fields=['main_slurm_args', 'merge_slurm_args', 'results_dir', 'input_data_file'])
        check_has_keys(self['main_slurm_args'], required_fields=['array', 'ntasks', 'mem-per-cpu', 'time', 'account'])
        check_has_keys(self['merge_slurm_args'], required_fields=['mem-per-cpu', 'account'])
        if 'copy' in self:
            check_has_keys(self['copy_slurm_args'], required_fields=['array', 'mem-per-cpu', 'time'])

    @property
    def array_elements(self):
        return parse_slurm_array(self['main_slurm_args']['array'])

    @property
    def array_size(self):
        return len(self.array_elements)
    
    @property
    def tmp_dir(self):
        return os.path.join(self['results_dir'], 'tmp')

    @property
    def batched_data_dir(self):
        return os.path.join(self.tmp_dir, 'data_batched')
    
    @property
    def batched_results_dir(self):
        return os.path.join(self.tmp_dir, 'results_batched')
    
    @property
    def split_results_dir(self):
        return os.path.join(self['results_dir'], 'split_results')
    
    @property
    def merged_results_file(self):
        return os.path.join(self['results_dir'], 'merged_results.txt')
    
    def setup(self):
        # Create directories
        os.makedirs(self['results_dir'], exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

        # Split data
        split_data(
            input_file=self['input_data_file'],
            batched_data_dir=self.batched_data_dir,
            array_ids=self.array_elements,
            num_proc_per_job=self['main_slurm_args']['ntasks']
        )

    def submit_main(self):
        self.main_job_id = self.main_job.submit()
    
    def submit_merge(self):
        if self.merge_job_id is not None:
            self.merge_job.dependencies = f"afterok:{self.main_job_id}"
        self.merge_job_id = self.merge_job.submit()
    
    def estimate_total_time(self, num_runs, single_run_time):
        estimate_total_time(
            num_runs=num_runs,
            single_run_time=single_run_time,
            job_array_size=self.array_size,
            n_tasks_per_job=self['main_slurm_args']['ntasks']
        )