import os
from typing import Union, Optional
from jinja2 import Template
import subprocess
from ..job import JobGroup
from ..utils import check_has_keys, remove_and_make_dir, submit_slurm_job, cancel_slurm_job, write_temp_file

script_template_content = \
"""#!/bin/bash -l

{% if is_array_job %}
#SBATCH --output={{ stdout_dir }}/slurm-%A_%a.out
{% else %}
#SBATCH --output={{ stdout_dir }}/slurm-%j.out
{% endif %}

module purge

# Set up CPU monitoring
# module load monitor
# monitor cpu percent > {{ resource_monitoring_dir }}/cpu-percent-run-$SLURM_JOB_ID{% if is_array_job %}_$SLURM_ARRAY_TASK_ID{% endif %}.log &
# CPU_USAGE_PID=$!
# monitor cpu memory > {{ resource_monitoring_dir }}/cpu-memory-run-$SLURM_JOB_ID{% if is_array_job %}_$SLURM_ARRAY_TASK_ID{% endif %}.log &
# CPU_MEM_PID=$!

# Set up GPU monitoring if requested
{% if use_gpu %}
nvidia-smi
module load monitor
monitor gpu percent > {{ resource_monitoring_dir }}/gpu-percent-run-$SLURM_JOB_ID{% if is_array_job %}_$SLURM_ARRAY_TASK_ID{% endif %}.log &
GPU_USAGE_PID=$!
monitor gpu memory > {{ resource_monitoring_dir }}/gpu-memory-run-$SLURM_JOB_ID{% if is_array_job %}_$SLURM_ARRAY_TASK_ID{% endif %}.log &
GPU_MEM_PID=$!
{% endif %}

# Run computations
# apptainer run {{ container_image }} {{ program }} \
# {% for key, value in program_args.items() if value is not none %}\
# {% if value is boolean and value %}--{{ key }} \
# {% elif value is not boolean %}--{{ key }}=\"{{ value }}\" \
# {% endif %}{% endfor %}
# apptainer run {{ container_image }} python3 {{ program }} \
# {% for key, value in program_args.items() if value is not none %}\
# {% if value is boolean and value %}--{{ key }} \
# {% elif value is not boolean %}--{{ key }}=\"{{ value }}\" \
# {% endif %}{% endfor %}
apptainer exec --nv --writable-tmpfs {{ container_image }} bash -c "
{{ prerun_commands }}
python3 {{ program }} \
{% for key, value in program_args.items() if value is not none %}\
{% if value is boolean and value %}--{{ key }} \
{% elif value is not boolean %}--{{ key }}=\"{{ value }}\" \
{% endif %}{% endfor %}
"


# Shut down the resource monitors
# kill -s INT $CPU_USAGE_PID $CPU_MEM_PID
# {% if use_gpu %}
# kill -s INT $GPU_USAGE_PID $GPU_MEM_PID
# {% endif %}
"""
script_template = Template(script_template_content)

class SingleJob(JobGroup):
    """A class for submitting a single job to a SLURM cluster."""
    def __init__(
        self,
        config: Union[str, dict, list[Union[str, dict, None]]]
    ):
        super().__init__(config)
        self.check_config_is_valid()

        print(self['program'])

        self.job_script = script_template.render(dict(
            container_image=self['container_image'],
            program=self['program'],
            program_args=self['program_args'],
            stdout_dir=self.stdout_dir,
            resource_monitoring_dir=self.resource_monitoring_dir,
            is_array_job = 'array' in self['slurm_args'].keys(),
            prerun_commands=self['prerun_commands'] if 'prerun_commands' in self.keys() else '',
        ))
        self.job_id = None
    
    def check_config_is_valid(self):
        check_has_keys(self, required_keys=['slurm_args', 'program', 'program_args', 'container_image'])
    
    @property
    def tmp_dir(self):
        if 'tmp' in self.keys():
            return self['tmp']
        else:
            return 'tmp'
    
    @property
    def job_scripts_dir(self):
        return os.path.join(self.tmp_dir, 'job_scripts')
    
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
    
    def _write_job_script(self, job_script_str):
        return write_temp_file(job_script_str, dir=self.job_scripts_dir, prefix='submit_')

    def setup(self, clear_directories: Optional[bool] = True):
        if clear_directories:
            remove_and_make_dir(self.tmp_dir)
        else:
            os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.job_scripts_dir, exist_ok=True)
        os.makedirs(self.resource_monitoring_dir, exist_ok=True)
        os.makedirs(self.stdout_dir, exist_ok=True)
    
    def submit(
        self, 
        dependency_ids: Optional[list[list[int]]] = None, 
        dependency_conditions: Optional[list[str]] = None,
        clear_directories: Optional[bool] = True,
        verbose: Optional[bool] = True
    ) -> tuple[int]:
        self.setup(clear_directories=clear_directories)
        job_script_filename = self._write_job_script(self.job_script)

        if verbose:
            print()
            print(' ------------')
            print('| SINGLE JOB |')
            print(' ------------')

        self.job_id = submit_slurm_job(
            slurm_args=self['slurm_args'], 
            job_script_filename=job_script_filename,
            dependency_ids=dependency_ids,
            dependency_conditions=dependency_conditions
        )

        if verbose:
            print(f"Diagnostics")
            print(f"-----------")
            if 'array' in self['slurm_args'].keys():
                print(f"Standard output/error files:    {os.path.abspath(self.stdout_dir)}/slurm-{self.job_id}_*.out")
            else:
                print(f"Standard output/error files:    {os.path.abspath(self.stdout_dir)}/slurm-{self.job_id}.out")
            print(f"Resource monitoring directory:  {os.path.abspath(self.resource_monitoring_dir)}")
            print()
        
        return (self.job_id,)

    
    def cancel(self):
        cancel_slurm_job(self.job_id)
        subprocess.run(["rm", "-rf", self.tmp_dir])