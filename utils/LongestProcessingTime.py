import pandas as pd
# modified from
# https://github.com/sanathkumarbs/longest-processing-time-algorithm-lpt

class LPT(object):
    """
    Implementation of LPT algorithm (Longest Processing Time). 
    Find the optimal minimal batch size to execune multiple videos with different legths
    
    """

    def __init__(self, df):
        """
        Input is a dataframe with the number of frames of each video in the column 'frame_num'
        """
        self.jobs = df
        self.max_processors = len(self.jobs)
        
        assert self.max_processors > 1, "Useless to run this algorithm for just one video" # denominator can't be 0 
        
        self.min_processors = 2
        
        self.processors = []
        self.scheduled_jobs = []
        self.loads = []
        self.max_load = []
        self.diff_max_load = []
        
        
    def find_best_batch_size(self):
        """
        Run the algorithm from 
        """
        
        for j,i in enumerate(range(self.min_processors,self.max_processors)):
            self.processors.append(i)
            temp_sj, temp_loads = self.lpt_algorithm_pd(i)
            self.scheduled_jobs.append(temp_sj)
            self.loads.append(temp_loads)
            self.max_load.append(max(temp_loads))
            if len(self.max_load)>1:
                diff_temp = self.max_load[j]-self.max_load[j-1]
                self.diff_max_load.append(diff_temp)
                if diff_temp>=0:
                    break
        return self.scheduled_jobs[-2], self.loads[-2]
    
    def run_for_custom_batch_size(self,custom_batch_size):
        """
        Run the algorithm from 
        """
        return self.lpt_algorithm_pd(custom_batch_size)

                
    @staticmethod
    def minloadproc(loads):
        """Find the processor with the minimum load.
        Break the tie of processors having same load on
        first come first serve basis.
        """
        minload = min(loads)
        for proc, load in enumerate(loads):
            if load == minload:
                return proc

    def lpt_algorithm_pd(self,number_procesors):

        loads = []
        scheduled_jobs = []
        for processor in range(number_procesors):
            loads.append(0)
            scheduled_jobs.append(pd.DataFrame())

        for i,row in self.jobs.iterrows():
            index_min_load = self.minloadproc(loads)
            scheduled_jobs[index_min_load] = scheduled_jobs[index_min_load].append(row)
            loads[index_min_load] += row.frame_num

        return scheduled_jobs, loads
