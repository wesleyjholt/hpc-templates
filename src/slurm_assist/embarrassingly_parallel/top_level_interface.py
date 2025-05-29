import os
import subprocess
from typing import Union, Optional
from collections import namedtuple
from jinja2 import Template
from . import split, run, merge
from .. import SingleJob
from ..job import JobGroup
from ..utils import (
    check_has_keys, 
    parse_field,
    parse_slurm_array, 
    estimate_total_time, 
    submit_slurm_job, 
    remove_and_make_dir, 
    cancel_slurm_job,
    write_temp_file,
    convert_slurm_keys,
    merge_dicts
)

from .. import utils
utils_parent_dir = os.path.dirname(os.path.abspath(utils.__file__))
split_python_script = os.path.abspath(split.__file__)
main_python_script = os.path.abspath(run.__file__)
merge_python_script = os.path.abspath(merge.__file__)

main_script_template_content = \
"""#!/bin/bash -l

#SBATCH --output={{ stdout_dir }}/slurm-%A_%a.out

module purge

# Set up CPU monitoring
module load monitor
monitor cpu percent > {{ resource_monitoring_dir }}/cpu-percent-run-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log &
CPU_USAGE_PID=$!
monitor cpu memory > {{ resource_monitoring_dir }}/cpu-memory-run-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log &
CPU_MEM_PID=$!

{% if use_gpu %}
# Set up GPU monitoring
module load monitor
monitor gpu percent > {{ resource_monitoring_dir }}/gpu-percent-run-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log &
GPU_USAGE_PID=$!
monitor gpu memory > {{ resource_monitoring_dir }}/gpu-memory-run-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log &
GPU_MEM_PID=$!
{% endif %}

# Run computations
# srun --mpi={{ mpi }} apptainer run {{ container_image }} {{ python_script }} --array-id $SLURM_ARRAY_TASK_ID {{ python_script_args }}
apptainer exec {% if use_gpu %}--nv {% endif %}--writable-tmpfs {{ container_image }} bash -c "
{{ prerun_commands }}  # pip install --no-cache-dir "../.."
python3 {{ python_script }} --array-id $SLURM_ARRAY_TASK_ID {{ python_script_args }}
"

# Shut down the resource monitors
kill -s INT $CPU_USAGE_PID $CPU_MEM_PID
{% if use_gpu %}
kill -s INT $GPU_USAGE_PID $GPU_MEM_PID
{% endif %}
"""
main_script_template = Template(main_script_template_content)

class EmbarrassinglyParallelJobs(JobGroup):
    def __init__(
        self, 
        config: Union[str, dict, list[Union[str, dict, None]]]
    ):
        super().__init__(config)
        print('======', self.keys())
        self._set_defaults()
        self.check_config_is_valid()

        # The "main job"
        main_python_script_args = [
            f"--single-run-module-parent-dir {self['single_run_module_parent_dir']}",
            f"--single-run-module {self['single_run_module']}",
            f"--single-run-fn {self['single_run_function']}",
            f"--batched-data-dir {self.batched_data_dir}",
            f"--batched-results-dir {self.batched_results_dir}",
            f"--split-results-dir {self.split_results_dir}",
            f"--job-array {self.main_slurm_args['array']}",
            f"--ntasks-per-job {self.main_slurm_args['ntasks']}",
            f"--utils-parent-dir {utils_parent_dir}",
        ]
        if '_main_python_script_extra_args' in self.keys():
            if isinstance(self['_main_python_script_extra_args'], dict):
                for k, v in self['_main_python_script_extra_args'].items():
                    main_python_script_args += [f"--{k} {v}"]
            else:
                raise ValueError(f"Expected '_main_python_script_extra_args' to be a dictionary, but got {type(self['_main_python_script_extra_args'])}.")
        main_python_script_args = ' '.join([parse_field(arg) for arg in main_python_script_args])
        self.main_job_script = main_script_template.render(dict(
            container_image=self['container_image'],
            python_script=main_python_script,
            python_script_args=main_python_script_args,
            mpi=self['mpi'],
            use_gpu=self['use_gpu'],
            stdout_dir=self.stdout_dir,
            resource_monitoring_dir=self.resource_monitoring_dir,
            prerun_commands=self['prerun_commands'],
        ))
        self.main_job_id = None

        self._default_split_merge_slurm_args = dict(
            time='00:10:00',
            mem='1G'
        )

    def add_suffix_to_job_name(self, slurm_args):
        if self.suffix is not None:
            slurm_args['job-name'] += f"_{self.suffix}"

    def check_config_is_valid(self, container_test_cmds=['python3', '-c', 'import os']):
        check_has_keys(self, required_keys=['main_slurm_args', 'results_dir', 'input_data_file', 'container_image', 'mpi', 'generate_new_ids'])
        if os.path.exists(self['container_image']):
            res = subprocess.run(['apptainer', 'exec', self['container_image'], *container_test_cmds])
            if res.returncode != 0:
                raise ValueError(f"Container image '{self['container_image']}' is not valid.")
        else:
            raise ValueError(f"Container image '{self['container_image']}' does not exist.")
    
    def check_input_data_file(self):
        if not os.path.isfile(self['input_data_file']):
            raise ValueError(f"Input data file '{self['input_data_file']}' is not a file.")
            
    def _set_defaults(self):
        if 'use_gpu' not in self:
            self['use_gpu'] = False
    
    @property
    def suffix(self):
        return self['job_name_suffix'] if 'job_name_suffix' in self.keys() else None

    @property
    def split_slurm_args(self):
        slurm_args = merge_dicts(
            convert_slurm_keys(self._default_split_merge_slurm_args),
            {'job-name': 'split'},
        )
        if 'split_slurm_args' in self.keys():
            slurm_args = merge_dicts(
                slurm_args,
                convert_slurm_keys(self['split_slurm_args'])
            )
        self.add_suffix_to_job_name(slurm_args)
        return slurm_args

    @property
    def main_slurm_args(self):
        slurm_args = merge_dicts(
            {
                'job-name': 'main',
                'use_gpu': False
            },
            convert_slurm_keys(self['main_slurm_args'])
        )
        self.add_suffix_to_job_name(slurm_args)
        return slurm_args
    
    @property
    def merge_slurm_args(self):
        slurm_args = merge_dicts(
            convert_slurm_keys(self._default_split_merge_slurm_args),
            {'job-name': 'merge'},
        )
        if 'merge_slurm_args' in self.keys():
            slurm_args = merge_dicts(
                slurm_args,
                convert_slurm_keys(self['merge_slurm_args'])
            )
        self.add_suffix_to_job_name(slurm_args)
        return slurm_args

    @property
    def array_elements(self):
        return parse_slurm_array(self.main_slurm_args['array'])

    @property
    def array_size(self):
        return len(self.array_elements)
    
    @property
    def tmp_dir(self):
        if 'tmp_dir' in self.keys():
            return os.path.relpath(self['tmp_dir'])
        else:
            return os.path.join(self['results_dir'], 'tmp')
    
    @property
    def job_scripts_dir(self):
        return os.path.join(self.tmp_dir, 'job_scripts')

    @property
    def batched_data_dir(self):
        return os.path.join(self.tmp_dir, 'data_batched')

    @property
    def batched_results_dir(self):
        return os.path.join(self.tmp_dir, 'results_batched')
    
    @property
    def log_dir(self):
        if 'log_dir' in self.keys():
            return os.path.relpath(self['log_dir'])
        else:
            return 'logs'
    
    @property
    def resource_monitoring_dir(self):
        return os.path.join(self.log_dir, 'resource_monitoring')
    
    @property
    def stdout_dir(self):
        return os.path.join(self.log_dir, 'stdout')
    
    @property
    def split_results_dir(self):
        return os.path.join(self['results_dir'], 'split_results')
    
    @property
    def merged_results_file(self):
        return os.path.join(self['results_dir'], 'merged_results.txt')
    
    def _write_job_script(self, job_script_str):
        return write_temp_file(job_script_str, dir=self.job_scripts_dir, prefix='submit_')
    
    def setup(self, clear_directories: Optional[bool] = True):
        # Main results directory
        if clear_directories:
            remove_and_make_dir(self['results_dir'])
            remove_and_make_dir(self.tmp_dir)
        else:
            os.makedirs(self['results_dir'], exist_ok=True)
            os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.job_scripts_dir, exist_ok=True)
        os.makedirs(self.batched_data_dir, exist_ok=True)
        os.makedirs(self.batched_results_dir, exist_ok=True)
        os.makedirs(self.split_results_dir, exist_ok=True)
        os.makedirs(self.resource_monitoring_dir, exist_ok=True)
        os.makedirs(self.stdout_dir, exist_ok=True)
    
    def submit_split(
        self,
        dependency_ids=None,
        dependency_conditions=None
    ):
        print("self.main_slurm_args['ntasks']: ", self.main_slurm_args['ntasks'])
        split_job = SingleJob(
            dict(
                program=split_python_script,
                program_args=dict(
                    utils_parent_dir=utils_parent_dir,
                    input_file=self['input_data_file'],
                    batched_data_dir=self.batched_data_dir,
                    job_array=self.main_slurm_args['array'],
                    ntasks_per_job=self.main_slurm_args['ntasks'],
                    generate_new_ids=self['generate_new_ids']
                ),
                tmp=self.tmp_dir,
                container_image=self['container_image'],
                slurm_args=self.split_slurm_args
            )
        )
        self.split_job_id = split_job.submit(
            dependency_ids=dependency_ids,
            dependency_conditions=dependency_conditions,
            clear_directories=False
        )[-1]  # Gets the last job ID (in this case, there is only one)

    def submit_main(
        self,
        dependency_ids=None,
        dependency_conditions=None
    ):
        job_script_filename = self._write_job_script(self.main_job_script)
        self.main_job_id = submit_slurm_job(
            slurm_args=self.main_slurm_args, 
            job_script_filename=job_script_filename, 
            dependency_ids=[[self.split_job_id]] if dependency_ids is None else dependency_ids, 
            dependency_conditions=['afterok'] if dependency_conditions is None else dependency_conditions,
        )

    def submit_merge(
        self,
        dependency_ids=None,
        dependency_conditions=None
    ):
        merge_job = SingleJob(
            dict(
                program=merge_python_script,
                program_args=dict(
                    utils_parent_dir=utils_parent_dir,
                    batched_results_dir=self.batched_results_dir,
                    merged_results_file=self.merged_results_file,
                    tmp_dir=self.tmp_dir,
                    job_array=self.main_slurm_args['array'],
                    ntasks_per_job=self.main_slurm_args['ntasks']
                ),
                tmp=self.tmp_dir,
                container_image=self['container_image'],
                slurm_args=self.merge_slurm_args
            )
        )
        self.merge_job_id = merge_job.submit(
            dependency_ids=[[self.main_job_id]] if dependency_ids is None else dependency_ids, 
            dependency_conditions=['afterany'] if dependency_conditions is None else dependency_conditions,
            clear_directories=False
        )[-1]  # Gets the last job ID (in this case, there is only one)

    
    def submit(
        self, 
        dependency_ids: Optional[list[list[int]]] = None, 
        dependency_conditions: Optional[list[str]] = None,
        clear_directories: Optional[bool] = True,
        verbose: Optional[bool] = True
    ) -> tuple[int]:
        """Submits the jobs. Returns ID of the last job(s) in the dependency chain."""
        if verbose:
            print()
            print(' ---------------')
            print('| PARALLEL JOBS |')
            print(' ---------------')
        
        self.setup(clear_directories=clear_directories)
        if not os.path.exists(self.batched_data_dir):
            raise ValueError(f"Batched data directory '{self.batched_data_dir}' does not exist.")
        
        if verbose:
            print()
            print('SPLIT JOB')
            print('=========')

        self.submit_split(dependency_ids=dependency_ids, dependency_conditions=dependency_conditions)
        
        if verbose:
            print()
            print('MAIN JOB')
            print('========')

        self.submit_main()

        if verbose:
            print(f"Diagnostics")
            print(f"-----------")
            print(f"Standard output/error files:    {os.path.abspath(self.stdout_dir)}/slurm-{self.main_job_id}_*.out")
            print(f"Resource monitoring directory:  {os.path.abspath(self.resource_monitoring_dir)}")
            print()

        if verbose:
            print()
            print('MERGE JOB')
            print('=========')

        self.submit_merge()

        JobIds = namedtuple('JobIds', ['split', 'main', 'merge'])
        return JobIds(split=self.split_job_id, main=self.main_job_id, merge=self.merge_job_id)
    
    def cancel(self):
        cancel_slurm_job(self.main_job_id)
        cancel_slurm_job(self.merge_job_id)
        subprocess.run(["rm", "-rf", self.tmp_dir])

    def estimate_total_time(self, num_runs, single_run_time):
        """Estimate total time for all jobs to complete (not including slurm queue wait times).

        Example
        -------

        estimate_total_time(
            1000000,  # total number of runs
            180  # time per run (in seconds)
        )
        """
        n_tasks_per_job = self.main_slurm_args['ntasks'] if self.main_slurm_args['ntasks'] is not None else 1
        estimate_total_time(
            num_runs=num_runs,
            single_run_time=single_run_time,
            job_array_size=self.array_size,
            n_tasks_per_job=n_tasks_per_job
        )