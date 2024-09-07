# Each job takes in config files and outputs results.
# Each transition takes in config files and outputs more config files.

from typing import Union, Callable, Optional, Sequence, Any
from ..job import JobGroup

ListLike = Union[list, tuple]
Config = Union[str, dict, list[Union[str, dict, None]]]
JobGroupGenFunc = Callable[[Config, Any], tuple[JobGroup, Any]]
ConfigGenFunc = Callable[[Config, Any], tuple[Config, Any]]
DependencyIds = tuple[int]
DependencyIdsList = list[DependencyIds]
DependencyCondList = list[str]
DependencyGenFunc = Callable[[DependencyIds, Any], tuple[tuple[DependencyIdsList, DependencyCondList], Any]]
State = Any

class SerialJobsWithState(JobGroup):
    """A collection of jobs to be run in serial.
    
    Parameters
    ----------
    config: Config
        Global configurations for the overall job group
    job_group_gen_fns: Union[JobGroupGenFunc, list[JobGroupGenFunc]]
        Job group generator functions. Takes as inputs a Config and a State, and returns
        a JobGroup (e.g., SingleJob, EmbarrassinglyParallelJobs) and a new State. Can
        be a single function or a list of such functions.
    config_gen_fns: Union[ConfigGenFunc, list[ConfigGenFunc]]
        Config object generator functions. Takes as inputs a Config and a State, and  
        returns a new Config and State. Can be a single function or a list of such 
        functions.
    dependency_gen_fns: Union[DependencyGenFunc, list[DependencyGenFunc]]
        Dependency object generator. Takes as inputs a DependencyIds and a State, 
        and returns a tuple (DependencyIds, DependencyConcList) and a State. Can be a 
        single function or a list of such functions.
    num_loops: Optional[int]
        The number of times to loop through submitting the jobs. Defaults to 1.
    state: Optional[State]
        An object containing any necessary information about the state of the serial jobs.
        This will be passed into job_group_gen_fns, config_gen_fns, and dependency_gen_fns.
    """
    def __init__(
        self,
        config: Config,
        job_group_gen_fns: Union[JobGroupGenFunc, list[JobGroupGenFunc]],
        config_gen_fns: Union[ConfigGenFunc, list[ConfigGenFunc]],
        dependency_gen_fns: Union[DependencyGenFunc, list[DependencyGenFunc]],
        num_loops: int = 1,
        state: Any = None
    ):
        super().__init__(config)
        self._config = config
        self.i_submit = 0
        if not isinstance(job_group_gen_fns, ListLike):
            job_group_gen_fns = [job_group_gen_fns]
        if not isinstance(config_gen_fns, ListLike):
            config_gen_fns = [config_gen_fns]
        if not isinstance(dependency_gen_fns, ListLike):
            dependency_gen_fns = [dependency_gen_fns]
        self.job_group_gen_fns, self.config_gen_fns, self.dependency_gen_fns = [], [], []
        for _ in range(num_loops):
            self.job_group_gen_fns += job_group_gen_fns
            self.config_gen_fns += config_gen_fns
            self.dependency_gen_fns += dependency_gen_fns
        self.config_gen_state = state
        self.job_groups = []
        self.last_job_ids = None
    
    def submit_next(self):
        config_i, self.config_gen_state = self.config_gen_fns[self.i_submit](self._config, self.config_gen_state)
        job_group_i, self.config_gen_state = self.job_group_gen_fns[self.i_submit](config_i, self.config_gen_state)
        if self.last_job_ids is not None:
            dep, self.config_gen_state = self.dependency_gen_fns[self.i_submit](self.last_job_ids, self.config_gen_state)
        else:
            dep = (None, None)
        self.last_job_ids = job_group_i.submit(dependency_ids=dep[0], dependency_conditions=dep[1])
        self.job_groups.append(job_group_i)
        self.i_submit += 1

    def submit(self):
        while self.i_submit < len(self.job_group_gen_fns):
            self.submit_next()


class SerialJobs(SerialJobsWithState):
    def __init__(
        self,
        job_groups = None,
        job_group_gen_fns = None,
        dependency_gen_fns = None,
        num_loops = None,
    ):
        